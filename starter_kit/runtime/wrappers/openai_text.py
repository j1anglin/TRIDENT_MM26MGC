from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from starter_kit.runtime.text import strip_modality_tags
from starter_kit.runtime.types import TaskSample
from starter_kit.runtime.wrappers.base import BaseModelWrapper, ModelCapabilities


class OpenAITextWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=False, supports_video=False, supports_audio=False)

    def __init__(
        self,
        *args: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout_s: float = 180.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.session: Optional[requests.Session] = None
        self.last_usage: Optional[Dict[str, Any]] = None

    def _load(self) -> None:
        if not self.api_key:
            raise ValueError(
                "OpenAI evaluator backend requires an API key. "
                "Pass --oeq-evaluator-api-key or set OPENAI_API_KEY."
            )
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        self.session = session

    def _build_input(self, prompt: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_hint.strip()}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )
        return messages

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        text = payload.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        output = payload.get("output")
        if not isinstance(output, list):
            return ""
        chunks: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text:
                    chunks.append(part_text)
        return "\n".join(chunks).strip()

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        assert self.session is not None

        if sample.modality not in {"text", "", None}:
            raise ValueError(f"OpenAITextWrapper only supports text modality, received: {sample.modality}")

        prompt = strip_modality_tags(sample.prompt or "").strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            prompt = f"{prefix}\n\n{prompt}" if prompt else prefix
        if not prompt:
            prompt = "Please answer the user query."

        self.last_usage = None
        payload = {
            "model": self.model_id,
            "input": self._build_input(prompt),
            "max_output_tokens": self.max_new_tokens,
        }
        response = self.session.post(
            f"{self.base_url}/responses",
            json=payload,
            timeout=self.request_timeout_s,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")

        data = response.json()
        usage = data.get("usage")
        if isinstance(usage, dict):
            self.last_usage = usage
        text = self._extract_text(data)
        if text:
            return text
        raise RuntimeError(f"OpenAI API returned no text output: {json.dumps(data, ensure_ascii=False)[:2000]}")
