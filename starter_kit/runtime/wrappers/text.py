from __future__ import annotations

from typing import Any, Dict, List

import torch

from starter_kit.runtime.text import strip_modality_tags
from starter_kit.runtime.torch import apply_dtype_kw
from starter_kit.runtime.types import TaskSample
from starter_kit.runtime.wrappers.base import BaseModelWrapper, ModelCapabilities


def _infer_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        try:
            return torch.device(model.device)
        except Exception:
            pass
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


class AutoTextWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=False, supports_video=False, supports_audio=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        load_kwargs.setdefault("device_map", "auto")
        cache_dir = load_kwargs.get("cache_dir")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs).eval()
        tokenizer_kwargs: Dict[str, Any] = {}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if hasattr(self.model, "config"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _build_prompt(self, prompt: str) -> str:
        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": self.system_hint.strip()})
        messages.append({"role": "user", "content": prompt})

        tokenizer = self.tokenizer
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        if self.system_hint:
            return f"{self.system_hint.strip()}\n\n{prompt}"
        return prompt

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        assert self.model is not None
        assert self.tokenizer is not None

        if sample.modality not in {"text", "", None}:
            raise ValueError(f"AutoTextWrapper only supports text modality, received: {sample.modality}")

        prompt = strip_modality_tags(sample.prompt or "").strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            prompt = f"{prefix}\n\n{prompt}" if prompt else prefix
        if not prompt:
            prompt = "Please answer the user query."

        full_prompt = self._build_prompt(prompt)
        model_inputs = self.tokenizer(full_prompt, return_tensors="pt")
        device = _infer_device(self.model)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.generation_overrides:
            gen_kwargs.update(self.generation_overrides)

        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **gen_kwargs)

        input_ids = model_inputs["input_ids"]
        continuation = output_ids[:, input_ids.shape[-1] :]
        response = self.tokenizer.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()
