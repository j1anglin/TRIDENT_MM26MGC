from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from starter_kit.runtime.torch import resolve_torch_dtype


@dataclass(frozen=True)
class ModelCapabilities:
    supports_image: bool = True
    supports_video: bool = True
    supports_audio: bool = False

    @property
    def supported_modalities(self) -> Tuple[str, ...]:
        modalities = []
        if self.supports_image:
            modalities.append("image")
        if self.supports_video:
            modalities.append("video")
        if self.supports_audio:
            modalities.append("audio")
        return tuple(modalities)


class BaseModelWrapper:
    capabilities: ModelCapabilities = ModelCapabilities()

    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int = 512,
        torch_dtype: Optional[torch.dtype] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._loaded = False
        self._torch_dtype = torch_dtype or resolve_torch_dtype("auto")
        self._load_kwargs = load_kwargs or {}
        self.system_hint: Optional[str] = None
        self.user_prefix: Optional[str] = None
        self.generation_overrides: Dict[str, Any] = {}
        self.preserve_whitespace: bool = False

    @property
    def supported_modalities(self) -> Iterable[str]:
        return self.capabilities.supported_modalities

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def generate(self, sample: Any) -> str:  # pragma: no cover
        raise NotImplementedError
