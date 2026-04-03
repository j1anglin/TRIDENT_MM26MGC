# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starter_kit.runtime.media import load_video_frames_raw
from starter_kit.runtime.text import strip_modality_tags
from starter_kit.runtime.torch import apply_dtype_kw
from starter_kit.runtime.types import TaskSample
from starter_kit.runtime.wrappers.base import BaseModelWrapper, ModelCapabilities

logger = logging.getLogger(__name__)


def _load_image(sample_path: str):
    from PIL import Image

    return Image.open(sample_path).convert("RGB")


def _load_audio_array(sample_path: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore

        data, sample_rate = sf.read(sample_path, always_2d=False)
    except ImportError as exc:
        raise ImportError(
            "Phi audio support requires the optional package `soundfile`. "
            "Install it (e.g., `pip install soundfile`) to enable audio evaluation."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load audio from {sample_path}: {exc}") from exc

    if isinstance(data, tuple):
        data, sample_rate = data  # type: ignore[assignment]
    data = np.asarray(data, dtype=np.float32)
    return data, sample_rate


def _move_to_device(
    batch: Dict[str, Any],
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if dtype is not None and torch.is_floating_point(value):
                value = value.to(dtype)
        moved[key] = value
    return moved


class _BasePhiWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True)
    placeholder_join: str = ""
    newline_after_placeholders: bool = False

    def __init__(
        self,
        *args,
        video_max_frames: int = 16,
        force_sliding_window: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.video_max_frames = video_max_frames
        self.num_frames = video_max_frames
        self._processor_kwargs: Dict[str, Any] = {}
        self.generation_config = None
        self.force_sliding_window = force_sliding_window

    def _load_audio(self, sample_path: str) -> tuple[np.ndarray, int]:
        audio, sample_rate = _load_audio_array(sample_path)
        return audio, int(sample_rate)

    def _load(self) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, GenerationConfig

        cache_dir = self._load_kwargs.get("cache_dir")
        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        load_kwargs.setdefault("trust_remote_code", True)
        attn_impl = load_kwargs.get("attn_implementation")
        if attn_impl:
            config_kwargs: Dict[str, Any] = {}
            if cache_dir:
                config_kwargs["cache_dir"] = cache_dir
            if load_kwargs.get("local_files_only"):
                config_kwargs["local_files_only"] = True
            config_kwargs["trust_remote_code"] = load_kwargs.get("trust_remote_code", True)
            try:
                config = AutoConfig.from_pretrained(self.model_id, **config_kwargs)
                current_attn = getattr(config, "_attn_implementation", None)
                if current_attn != attn_impl:
                    setattr(config, "_attn_implementation", attn_impl)
                    text_cfg = getattr(config, "text_config", None)
                    if text_cfg is not None:
                        try:
                            setattr(text_cfg, "_attn_implementation", attn_impl)
                        except Exception:
                            logger.debug("Unable to sync text_config._attn_implementation.", exc_info=True)
                load_kwargs["config"] = config
            except Exception:
                logger.warning(
                    "Failed to preload Phi config for attn_implementation=%s; continuing with defaults.", attn_impl
                )

        model_id_lower = str(self.model_id).lower()
        if "phi-3.5" in model_id_lower and "vision" in model_id_lower:
            load_kwargs["trust_remote_code"] = True
            try:
                from transformers import CLIPVisionModel

                fwd = getattr(CLIPVisionModel, "forward", None)
                if callable(fwd) and not getattr(fwd, "_compat_accepts_kwargs", False):
                    sig = inspect.signature(fwd)
                    has_varkw = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
                    if not has_varkw:
                        original_forward = fwd

                        def _clip_forward(
                            self,
                            pixel_values=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=None,
                            **kwargs,
                        ):
                            return original_forward(
                                self,
                                pixel_values=pixel_values,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                            )

                        _clip_forward.__doc__ = getattr(original_forward, "__doc__", None)
                        _clip_forward._compat_accepts_kwargs = True  # type: ignore[attr-defined]
                        setattr(CLIPVisionModel, "forward", _clip_forward)
                        logger.warning("Injected CLIPVisionModel.forward(**kwargs) compatibility shim.")
            except Exception:
                logger.debug("CLIPVisionModel forward compatibility patch skipped.", exc_info=True)

        def _ensure_phi4mm_prepare_inputs_hook() -> None:
            if os.environ.get("PHI4_DISABLE_PEFT_PATCH", "").strip():
                logger.info("PHI4_DISABLE_PEFT_PATCH set; skip PEFT monkey-patch.")
                return
            try:
                import peft.mapping_func as peft_mapping  # type: ignore
            except Exception:
                logger.debug("PEFT not available; skip PEFT monkey-patch.")
                return
            if getattr(_ensure_phi4mm_prepare_inputs_hook, "_patched", False):
                return

            original_get_peft_model = peft_mapping.get_peft_model

            def patched_get_peft_model(model, peft_config, adapter_name: str = "default", **kwargs):
                cls = model.__class__
                mt = getattr(getattr(model, "config", None), "model_type", "")
                if (not hasattr(cls, "prepare_inputs_for_generation")) and ("phi4" in (mt or "").lower()):

                    def _prepare_inputs_for_generation(
                        self,
                        input_ids,
                        past_key_values=None,
                        attention_mask=None,
                        inputs_embeds=None,
                        input_image_embeds=None,
                        image_sizes=None,
                        image_attention_mask=None,
                        input_audio_embeds=None,
                        audio_embed_sizes=None,
                        audio_attention_mask=None,
                        input_mode=None,
                        cache_position=None,
                        position_ids=None,
                        use_cache=True,
                        num_logits_to_keep=None,
                        **inner_kwargs,
                    ):
                        if num_logits_to_keep is None:
                            num_logits_to_keep = 1

                        if (
                            past_key_values
                            and getattr(self.config, "rope_scaling", None)
                            and input_ids.shape[1] >= getattr(self.config, "original_max_position_embeddings", 0) + 1
                        ):
                            past_length = cache_position[0] if cache_position is not None else 0
                            if past_length <= getattr(self.config, "original_max_position_embeddings", 0):
                                past_key_values = None

                        return super(cls, self).prepare_inputs_for_generation(  # type: ignore[misc]
                            input_ids=input_ids,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            inputs_embeds=inputs_embeds,
                            input_image_embeds=input_image_embeds,
                            image_sizes=image_sizes,
                            image_attention_mask=image_attention_mask,
                            input_audio_embeds=input_audio_embeds,
                            audio_embed_sizes=audio_embed_sizes,
                            audio_attention_mask=audio_attention_mask,
                            input_mode=input_mode,
                            cache_position=cache_position,
                            position_ids=position_ids,
                            use_cache=use_cache,
                            num_logits_to_keep=num_logits_to_keep,
                            **inner_kwargs,
                        )

                    setattr(cls, "prepare_inputs_for_generation", _prepare_inputs_for_generation)

                return original_get_peft_model(model, peft_config, adapter_name=adapter_name, **kwargs)

            peft_mapping.get_peft_model = patched_get_peft_model
            try:
                import peft  # type: ignore

                peft.get_peft_model = patched_get_peft_model  # type: ignore[attr-defined]
            except Exception:
                pass

            try:
                from transformers.cache_utils import Cache

                if not hasattr(Cache, "get_usable_length"):

                    def _cache_get_usable_length(self, *args, **kwargs):  # type: ignore[override]
                        new_seq_len = None
                        layer_idx = None
                        if len(args) >= 1:
                            new_seq_len = args[0]
                        if len(args) >= 2:
                            layer_idx = args[1]
                        if new_seq_len is None:
                            new_seq_len = kwargs.get("new_seq_len", 0)
                        if layer_idx is None:
                            layer_idx = kwargs.get("layer_idx", None)
                        try:
                            layers = getattr(self, "layers", [])
                            if not layers:
                                return 0

                            def usable_for_layer(idx: int) -> int:
                                if idx < 0 or idx >= len(layers):
                                    return 0
                                length = layers[idx].get_seq_length()
                                max_cache = layers[idx].get_max_cache_shape()
                                if isinstance(max_cache, int) and max_cache > 0:
                                    length = min(length, max_cache)
                                return int(length)

                            if layer_idx is None:
                                vals = [usable_for_layer(i) for i in range(len(layers))]
                                return min(vals) if vals else 0
                            return usable_for_layer(int(layer_idx))
                        except Exception:
                            return 0

                    setattr(Cache, "get_usable_length", _cache_get_usable_length)
                    logger.warning("Injected fallback Cache.get_usable_length (varargs).")
            except Exception:
                logger.debug("Cache fallback injection skipped.", exc_info=True)

            try:
                from transformers.cache_utils import DynamicCache

                if not hasattr(DynamicCache, "seen_tokens"):

                    def _dc_get_seen_tokens(self):
                        try:
                            return self.get_seq_length()
                        except Exception:
                            return 0

                    def _dc_set_seen_tokens(self, value):
                        try:
                            self._compat_seen_tokens = int(value) if value is not None else 0
                        except Exception:
                            self._compat_seen_tokens = 0

                    DynamicCache.seen_tokens = property(_dc_get_seen_tokens, _dc_set_seen_tokens)
                    logger.warning("Injected DynamicCache.seen_tokens -> get_seq_length().")

                if not hasattr(DynamicCache, "get_max_length"):

                    def _dc_get_max_length(self):
                        try:
                            maxes = []
                            for layer in getattr(self, "layers", []):
                                try:
                                    max_cache = layer.get_max_cache_shape()
                                    if isinstance(max_cache, int) and max_cache > 0:
                                        maxes.append(max_cache)
                                except Exception:
                                    pass
                            if maxes:
                                return min(maxes)
                        except Exception:
                            pass
                        try:
                            cfg = getattr(self, "config", None)
                            sliding_window = None
                            if cfg is not None:
                                sliding_window = getattr(cfg, "sliding_window", None)
                                if sliding_window is None:
                                    text_cfg = getattr(cfg, "text_config", None)
                                    sliding_window = getattr(text_cfg, "sliding_window", None) if text_cfg else None
                            if isinstance(sliding_window, int) and sliding_window > 0:
                                return sliding_window
                        except Exception:
                            pass
                        return 0

                    setattr(DynamicCache, "get_max_length", _dc_get_max_length)
                    logger.warning("Injected DynamicCache.get_max_length().")

                if not hasattr(DynamicCache, "max_cache_length"):
                    DynamicCache.max_cache_length = property(lambda self: self.get_max_length())

                if not hasattr(DynamicCache, "from_legacy_cache"):

                    def _dc_from_legacy_cache(cls, past_key_values=None, num_hidden_layers=None):
                        cache = cls()
                        if past_key_values is not None:
                            for layer_idx, layer_cache in enumerate(past_key_values):
                                if not layer_cache:
                                    continue
                                key_states, value_states = layer_cache[:2]
                                cache.update(key_states, value_states, layer_idx)
                        return cache

                    DynamicCache.from_legacy_cache = classmethod(_dc_from_legacy_cache)
                    logger.warning("Injected DynamicCache.from_legacy_cache().")

                if not hasattr(DynamicCache, "to_legacy_cache"):

                    def _dc_to_legacy_cache(self):
                        legacy_cache = ()
                        try:
                            for layer in getattr(self, "layers", []):
                                if layer is None:
                                    continue
                                key_states = getattr(layer, "keys", None)
                                value_states = getattr(layer, "values", None)
                                if key_states is None or value_states is None:
                                    continue
                                legacy_cache += ((key_states, value_states),)
                        except Exception:
                            logger.debug("DynamicCache.to_legacy_cache fallback failed.", exc_info=True)
                        return legacy_cache

                    DynamicCache.to_legacy_cache = _dc_to_legacy_cache
                    logger.warning("Injected DynamicCache.to_legacy_cache().")
            except Exception:
                logger.debug("DynamicCache compatibility patch skipped.", exc_info=True)

            _ensure_phi4mm_prepare_inputs_hook._patched = True
            logger.info("PEFT monkey-patch for Phi4 applied (plus cache compat).")

        _ensure_phi4mm_prepare_inputs_hook()

        def _load_fallback_model() -> Optional[torch.nn.Module]:
            if "phi-3.5" in model_id_lower and "vision" in model_id_lower:
                return None
            try:
                from transformers.models.phi4_multimodal import Phi4MultimodalForCausalLM  # type: ignore

                logger.info("Loading Phi4MultimodalForCausalLM via transformers.")
                return Phi4MultimodalForCausalLM.from_pretrained(self.model_id, **load_kwargs).eval()
            except (ImportError, AttributeError):
                pass
            except Exception:
                logger.exception("Transformers Phi4MultimodalForCausalLM load failed.")
                return None
            try:
                remote_module = import_module("modeling_phi4mm")
                base_cls = getattr(remote_module, "Phi4MMModel", None)
                if base_cls is not None and not hasattr(base_cls, "prepare_inputs_for_generation"):

                    def _base_prepare_inputs_for_generation(
                        self,
                        input_ids,
                        past_key_values=None,
                        attention_mask=None,
                        inputs_embeds=None,
                        input_image_embeds=None,
                        image_sizes=None,
                        image_attention_mask=None,
                        input_audio_embeds=None,
                        audio_embed_sizes=None,
                        audio_attention_mask=None,
                        input_mode=None,
                        cache_position=None,
                        position_ids=None,
                        use_cache=True,
                        num_logits_to_keep=None,
                        **kwargs,
                    ):
                        if num_logits_to_keep is None:
                            num_logits_to_keep = 1
                        if (
                            past_key_values
                            and getattr(self.config, "rope_scaling", None)
                            and input_ids.shape[1] >= getattr(self.config, "original_max_position_embeddings", 0) + 1
                        ):
                            past_length = cache_position[0] if cache_position is not None else 0
                            if past_length <= getattr(self.config, "original_max_position_embeddings", 0):
                                past_key_values = None

                        return super(base_cls, self).prepare_inputs_for_generation(  # type: ignore[misc]
                            input_ids=input_ids,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            inputs_embeds=inputs_embeds,
                            input_image_embeds=input_image_embeds,
                            image_sizes=image_sizes,
                            image_attention_mask=image_attention_mask,
                            input_audio_embeds=input_audio_embeds,
                            audio_embed_sizes=audio_embed_sizes,
                            audio_attention_mask=audio_attention_mask,
                            input_mode=input_mode,
                            cache_position=cache_position,
                            position_ids=position_ids,
                            use_cache=use_cache,
                            num_logits_to_keep=num_logits_to_keep,
                            **kwargs,
                        )

                    setattr(base_cls, "prepare_inputs_for_generation", _base_prepare_inputs_for_generation)
                cls = getattr(remote_module, "Phi4MMForCausalLM", None)
                if cls is not None:
                    logger.info("Loading Phi4MMForCausalLM via remote module.")
                    return cls.from_pretrained(self.model_id, **load_kwargs).eval()
            except Exception:
                logger.exception("Remote Phi4MMForCausalLM load failed.")
                return None
            return None

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs).eval()
            logger.info("Loaded model via AutoModelForCausalLM.")
        except Exception:
            logger.exception("AutoModelForCausalLM load failed; trying fallback.")
            fallback = _load_fallback_model()
            if fallback is None:
                raise
            self.model = fallback
        else:
            if not hasattr(self.model, "prepare_inputs_for_generation"):
                logger.warning("Loaded model lacks prepare_inputs_for_generation; trying fallback.")
                fallback = _load_fallback_model()
                if fallback is None:
                    raise RuntimeError(
                        "Failed to load a generative Phi-4 model; no prepare_inputs_for_generation implementation found."
                    )
                self.model = fallback

        try:
            self._normalize_sliding_window(force_value=self.force_sliding_window)
        except Exception:
            logger.exception("Sliding-window normalization failed; continuing with defaults.")

        processor_kwargs = dict(self._processor_kwargs)
        if cache_dir:
            processor_kwargs.setdefault("cache_dir", cache_dir)
        processor_kwargs.setdefault("trust_remote_code", True)
        if "phi-3.5" in model_id_lower and "vision" in model_id_lower:
            processor_kwargs["trust_remote_code"] = True
        self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)

        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_id, cache_dir=cache_dir)
        except Exception:
            logger.debug("GenerationConfig not found; using None.")
            self.generation_config = None

    def _normalize_sliding_window(self, force_value: Optional[int] = None) -> None:
        cfg = getattr(self.model, "config", None)

        def _coerce_int(val):
            if val is None:
                return None
            if isinstance(val, int):
                return val
            try:
                parsed = float(val)
                if parsed < 0:
                    return None
                return int(parsed)
            except Exception:
                return None

        def _get_text_cfg_from(cfg_obj):
            if cfg_obj is None:
                return None
            text_cfg = getattr(cfg_obj, "text_config", None)
            if text_cfg is not None:
                return text_cfg
            get_text_config = getattr(cfg_obj, "get_text_config", None)
            if callable(get_text_config):
                try:
                    return get_text_config(decoder=True)
                except TypeError:
                    try:
                        return get_text_config()
                    except Exception:
                        return None
            return None

        if cfg is None:
            logger.warning("Model has no config; cannot normalize sliding_window.")
            return

        text_cfg = _get_text_cfg_from(cfg)

        if force_value is not None:
            sliding_window = max(0, int(force_value))
            logger.warning("Forcing sliding_window to %d via force_sliding_window.", sliding_window)
        else:
            candidates = [
                getattr(cfg, "sliding_window", None),
                getattr(text_cfg, "sliding_window", None) if text_cfg is not None else None,
                getattr(cfg, "sliding_window_size", None),
                getattr(text_cfg, "sliding_window_size", None) if text_cfg is not None else None,
                getattr(cfg, "window_size", None),
                getattr(text_cfg, "window_size", None) if text_cfg is not None else None,
            ]
            sliding_window = None
            for candidate in candidates:
                sliding_window = _coerce_int(candidate)
                if sliding_window is not None:
                    break
            if sliding_window is None:
                sliding_window = 0
                logger.warning("sliding_window missing; defaulting to 0 (disabled).")

        max_pos = getattr(cfg, "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0 and sliding_window > max_pos:
            logger.warning("sliding_window=%d > max_position_embeddings=%d; clamping.", sliding_window, max_pos)
            sliding_window = max_pos

        def _try_set(obj, name, value):
            if obj is None:
                return
            try:
                if getattr(obj, name, None) is None:
                    setattr(obj, name, value)
            except Exception:
                pass

        try:
            if getattr(cfg, "sliding_window", None) != sliding_window:
                setattr(cfg, "sliding_window", sliding_window)
        except Exception:
            pass

        _try_set(text_cfg, "sliding_window", sliding_window)
        for obj in (cfg, text_cfg):
            _try_set(obj, "sliding_window_size", sliding_window)
            _try_set(obj, "window_size", sliding_window)

        logger.info("Normalized sliding_window to %d", sliding_window)

    def _prepare_images(self, sample: TaskSample) -> List[Any]:
        if sample.modality == "image":
            return [_load_image(sample.fake_path)]
        return load_video_frames_raw(
            Path(sample.fake_path),
            num_segments=self.video_max_frames,
            max_frames=self.video_max_frames,
        )

    def _build_user_content(self, prompt: str, image_count: int, audio_count: int = 0) -> str:
        token_blocks: List[str] = []
        if image_count > 0:
            token_blocks.extend(f"<|image_{idx + 1}|>" for idx in range(image_count))
        if audio_count > 0:
            token_blocks.extend(f"<|audio_{idx + 1}|>" for idx in range(audio_count))

        if not token_blocks:
            return prompt

        placeholder_block = self.placeholder_join.join(token_blocks)
        if self.newline_after_placeholders and placeholder_block:
            placeholder_block += "\n"
        if not prompt:
            return placeholder_block
        if placeholder_block:
            return f"{placeholder_block}{prompt}"
        return prompt

    def _prepare_processor_inputs(
        self,
        prompt: str,
        images: List[Any],
        audios: Optional[List[Any]] = None,
    ):
        image_payload: Any = None
        if images:
            image_payload = images if len(images) > 1 else images[0]

        processor_kwargs: Dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        if image_payload is not None:
            processor_kwargs["images"] = image_payload
        if audios:
            processor_kwargs["audios"] = audios

        try:
            return self.processor(**processor_kwargs)
        except TypeError:
            call_kwargs = {"return_tensors": "pt"}
            if image_payload is None and not audios:
                try:
                    return self.processor(prompt, **call_kwargs)
                except TypeError:
                    return self.processor(text=prompt, **call_kwargs)
            if image_payload is not None and not audios:
                try:
                    return self.processor(prompt, image_payload, **call_kwargs)
                except TypeError:
                    return self.processor(text=prompt, images=image_payload, **call_kwargs)
            if audios and image_payload is None:
                audio_payload = audios if len(audios) > 1 else audios[0]
                try:
                    return self.processor(prompt, audio_payload, **call_kwargs)
                except TypeError:
                    return self.processor(text=prompt, audios=audios, **call_kwargs)
            audio_payload = audios if audios and len(audios) > 1 else (audios[0] if audios else None)
            return self.processor(prompt, image_payload, audio_payload, **call_kwargs)

    def _supports_kwarg(self, name: str) -> bool:
        try:
            for fn_name in ("generate", "prepare_inputs_for_generation"):
                fn = getattr(self.model, fn_name, None)
                if fn is None:
                    continue
                sig = inspect.signature(fn)
                if name in sig.parameters:
                    return True
        except Exception:
            pass
        return False

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()

        prompt_body = strip_modality_tags(sample.prompt).strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            prompt_body = f"{prefix}\n\n{prompt_body}" if prompt_body else prefix

        audios: Optional[List[Any]] = None
        if sample.modality == "audio":
            audio_data = self._load_audio(sample.fake_path)
            audios = [audio_data]
            images: List[Any] = []
        else:
            images = self._prepare_images(sample)

        user_text = self._build_user_content(prompt_body, len(images), len(audios) if audios else 0)

        messages: List[Dict[str, str]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": self.system_hint})
        messages.append({"role": "user", "content": user_text})

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt.endswith("<|endoftext|>"):
            prompt = prompt[: -len("<|endoftext|>")]

        processor_inputs = self._prepare_processor_inputs(prompt, images, audios=audios)
        device = getattr(self.model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = getattr(self.model, "dtype", None)
        inputs = _move_to_device(dict(processor_inputs.items()), device, dtype)
        input_length = inputs["input_ids"].shape[-1]

        gen_args: Dict[str, Any] = dict(max_new_tokens=self.max_new_tokens, do_sample=False, temperature=0.0)
        gen_args.update(self.generation_overrides or {})
        if self.generation_config is not None and "generation_config" not in gen_args:
            gen_args["generation_config"] = self.generation_config

        if self._supports_kwarg("num_logits_to_keep"):
            if gen_args.get("num_logits_to_keep") is None:
                gen_args["num_logits_to_keep"] = 1
        else:
            gen_args.pop("num_logits_to_keep", None)

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_args)

        sequences = output.sequences if hasattr(output, "sequences") else output
        generated = sequences[:, input_length:]
        text = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text.strip()


class Phi4MultimodalWrapper(_BasePhiWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True, supports_audio=True)
    placeholder_join = ""
    newline_after_placeholders = False

    def __init__(
        self,
        *args,
        video_max_frames: int = 16,
        force_sliding_window: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, video_max_frames=video_max_frames, force_sliding_window=force_sliding_window, **kwargs)
