from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class TaskSample:
    task: str
    sample_id: str
    modality: str
    prompt: str
    fake_path: str
    relative_fake_path: str
    label: str = "fake"
    file_sha256: Optional[str] = None
    media_meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fake_path"] = str(payload["fake_path"])
        payload["relative_fake_path"] = str(payload["relative_fake_path"])
        return payload


@dataclass
class ModelResponse:
    model_id: str
    sample: TaskSample
    response: str
    latency_ms: float
    fallback_count: int = 0
    final_seed: int = 0
    system_hint: Optional[str] = None
    usage_metadata: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "sample": self.sample.to_json(),
            "response": self.response,
            "timestamp": time.time(),
            "fallback_count": self.fallback_count,
            "final_seed": self.final_seed,
            "system_hint": self.system_hint,
            "usage_metadata": self.usage_metadata,
        }
