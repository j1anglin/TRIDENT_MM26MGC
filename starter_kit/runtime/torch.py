from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch


def resolve_torch_dtype(dtype_str: str = "auto") -> torch.dtype:
    s = (dtype_str or "auto").lower()
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    if s == "fp32":
        return torch.float32
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
        except Exception:
            major = 0
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def parse_visible_gpus(gpu_arg: Optional[str]) -> List[int]:
    if gpu_arg:
        return [int(x.strip()) for x in gpu_arg.split(",") if x.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        return list(range(len([x for x in env.split(",") if x.strip()])))
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def _flash_attn_supported() -> Tuple[bool, str]:
    if os.environ.get("FORCE_FLASH_ATTN", "").lower() in {"1", "true", "yes"}:
        return True, ""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:  # noqa: BLE001
        return False, f"unable to query CUDA capability ({exc})"
    if major < 8:
        return False, f"compute capability {major}.{minor} < 8.0"
    try:
        import flash_attn  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return False, "flash-attn package is not installed"
    except Exception as exc:  # noqa: BLE001
        return False, f"flash-attn import failed ({exc})"
    return True, ""


def build_loading_kwargs(
    device_map: str,
    gpus: Optional[str],
    per_gpu_max_memory_gib: Optional[int],
    flash_attn: bool,
    cache_dir: Optional[str],
    offline: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"device_map": device_map, "low_cpu_mem_usage": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if offline:
        kwargs["local_files_only"] = True
    visible = parse_visible_gpus(gpus)
    if device_map != "cpu" and visible and per_gpu_max_memory_gib:
        kwargs["max_memory"] = {i: f"{per_gpu_max_memory_gib}GiB" for i in visible}
        kwargs["max_memory"]["cpu"] = "32GiB"
    if flash_attn:
        supported, reason = _flash_attn_supported()
        if supported:
            kwargs["attn_implementation"] = "flash_attention_2"
        else:
            print(f"[WARN] flash-attn requested but unavailable: {reason}. Falling back to standard attention.")
            kwargs["attn_implementation"] = "sdpa"
    else:
        kwargs["attn_implementation"] = kwargs.get("attn_implementation", "sdpa")
    return kwargs


def apply_dtype_kw(load_kwargs: Optional[dict], dtype: torch.dtype) -> dict:
    kw = dict(load_kwargs or {})
    kw.pop("dtype", None)
    kw.setdefault("torch_dtype", dtype)
    return kw


def load_auto_processor(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None,
):
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, use_fast=True, cache_dir=cache_dir
        )
    except TypeError:
        return AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, cache_dir=cache_dir
        )
    except Exception:
        return AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, cache_dir=cache_dir
        )
