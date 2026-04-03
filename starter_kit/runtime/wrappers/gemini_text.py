from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from starter_kit.runtime.text import strip_modality_tags
from starter_kit.runtime.types import TaskSample
from starter_kit.runtime.wrappers.base import BaseModelWrapper, ModelCapabilities


class GeminiTextWrapper(BaseModelWrapper):
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
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.session: Optional[requests.Session] = None
        self.last_usage: Optional[Dict[str, Any]] = None

    def _load(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Gemini evaluator backend requires an API key. "
                "Pass --oeq-evaluator-api-key or set GEMINI_API_KEY / GOOGLE_API_KEY."
            )
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        self.session = session

    def _build_prompt(self, prompt: str) -> str:
        if self.system_hint:
            return f"{self.system_hint.strip()}\n\n{prompt}"
        return prompt

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return ""
        chunks: List[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
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
            raise ValueError(f"GeminiTextWrapper only supports text modality, received: {sample.modality}")

        prompt = strip_modality_tags(sample.prompt or "").strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            prompt = f"{prefix}\n\n{prompt}" if prompt else prefix
        if not prompt:
            prompt = "Please answer the user query."
        prompt = self._build_prompt(prompt)

        self.last_usage = None
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self.max_new_tokens,
            },
        }
        response = self.session.post(
            f"{self.base_url}/models/{self.model_id}:generateContent",
            params={"key": self.api_key},
            json=payload,
            timeout=self.request_timeout_s,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

        data = response.json()
        usage = data.get("usageMetadata")
        if isinstance(usage, dict):
            self.last_usage = usage
        text = self._extract_text(data)
        if text:
            return text
        raise RuntimeError(f"Gemini API returned no text output: {json.dumps(data, ensure_ascii=False)[:2000]}")
