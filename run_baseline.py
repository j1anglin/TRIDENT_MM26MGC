from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import gc
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import numpy as np
import torch

from starter_kit._bootstrap import DEFAULT_DATA_ROOT, DEFAULT_MODEL_CACHE_DIR, DEFAULT_OUTPUT_ROOT
from starter_kit.blind_data import (
    BlindBatch,
    expand_tasks,
    load_batches,
    normalize_modalities,
    task_info,
    task_system_hint,
)
from starter_kit.runtime.media import require_video_support
from starter_kit.runtime.text import PROMPT_ECHO_RE, canonicalize_text_block, clean_prompt_for_echo
from starter_kit.runtime.torch import build_loading_kwargs, resolve_torch_dtype
from starter_kit.runtime.wrappers.phi import Phi4MultimodalWrapper
from starter_kit.runtime.wrappers.qwen import Qwen3VLWrapper


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_ALIASES = {
    "qwen3-vl": DEFAULT_MODEL,
    "qwen3-vl-8b": DEFAULT_MODEL,
    "qwen3-vl-8b-instruct": DEFAULT_MODEL,
    "phi4": "microsoft/Phi-4-multimodal-instruct",
    "phi-4": "microsoft/Phi-4-multimodal-instruct",
    "phi4-mm": "microsoft/Phi-4-multimodal-instruct",
    "phi-4-mm": "microsoft/Phi-4-multimodal-instruct",
    "phi4-multimodal": "microsoft/Phi-4-multimodal-instruct",
    "phi-4-multimodal": "microsoft/Phi-4-multimodal-instruct",
    "phi-4-multimodal-instruct": "microsoft/Phi-4-multimodal-instruct",
}
REFUSAL_PATTERNS = [
    r"\bi'?m\s*sorry\b.*\b(can'?t|cannot)\s*(assist|help|comply)\b",
    r"\b(cannot|can't)\s*(assist|help|comply)\b",
    r"\bi\s*can't\s*assist\s*with\s*that\s*request\b",
    r"\bi\s*cannot\s*assist\s*with\s*that\s*request\b",
    r"\bI\s*can(?:not|'t)\s*help\s*with\s*that\b",
    r"\bi'?m\s*sorry\b.*\b(can'?t|cannot)\s*provide\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal starter-kit baseline on the blind competition package."
    )
    parser.add_argument(
        "--task",
        default="all",
        help="One of: typeb_oeq (Type-B OEQ), typea_oeq (Type-A OEQ), mcq, tfq, all",
    )
    parser.add_argument("--split", default="public_val", choices=["train", "public_val", "private_test"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--modalities", nargs="+", default=["image", "video", "audio"])
    parser.add_argument("--question-files", nargs="+", default=None, help="MCQ/TFQ only")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential", "cuda", "cpu"],
    )
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--per-gpu-max-memory-gib", type=int, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--video-max-frames", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_MODEL_CACHE_DIR))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _existing_ids(path: Path) -> Optional[Set[str]]:
    ids: Set[str] = set()
    if not path.exists():
        return ids
    try:
        with path.open("r+", encoding="utf-8") as handle:
            last_good = handle.tell()
            while True:
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    last_good = handle.tell()
                    continue
                record = json.loads(stripped)
                record_id = str(record.get("record_id") or "").strip()
                if not record_id:
                    handle.seek(last_good)
                    handle.truncate()
                    break
                ids.add(record_id)
                last_good = handle.tell()
    except Exception:
        return None
    return ids


def sanitize_model_id(model_id: str) -> str:
    raw = str(model_id or "").rstrip("/")
    for part in reversed(Path(raw).parts):
        if part.startswith("models--"):
            tokens = [token for token in part.split("--") if token]
            if len(tokens) >= 3:
                return tokens[-1].replace(":", "_")
    base = raw.split("/")[-1]
    return base.replace(":", "_")


def maybe_localize(model_id: str, cache_dir: str) -> str:
    p1 = os.path.join(cache_dir, model_id)
    p2 = os.path.join(cache_dir, model_id.split("/")[-1])
    return p1 if os.path.isdir(p1) else (p2 if os.path.isdir(p2) else model_id)


def resolve_model_alias(model_id: str) -> str:
    key = str(model_id or "").strip()
    return MODEL_ALIASES.get(key.lower(), key)


def _format_exception_chain(exc: BaseException) -> str:
    segments = []
    seen: Set[int] = set()
    current: Optional[BaseException] = exc
    while current and id(current) not in seen:
        seen.add(id(current))
        segments.append(f"{current.__class__.__name__}: {current}")
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        if current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
            continue
        break
    return " -> ".join(segments)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _cleanup_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _is_oom_error(error_msg: Optional[str]) -> bool:
    if not error_msg:
        return False
    normalized = error_msg.lower()
    return (
        "outofmemoryerror" in normalized
        or "cuda out of memory" in normalized
        or "hip out of memory" in normalized
        or "out of memory" in normalized
    )


def _apply_memory_retry_overrides(
    *,
    wrapper,
    sample,
    attempt: int,
    base_max_new_tokens: int,
    base_video_max_frames: Optional[int],
) -> Optional[str]:
    changes: list[str] = []

    if base_max_new_tokens > 0:
        retry_token_cap = 128 if attempt == 1 else 64
        capped_tokens = min(base_max_new_tokens, retry_token_cap)
        if capped_tokens < base_max_new_tokens:
            wrapper.generation_overrides["max_new_tokens"] = capped_tokens
            changes.append(f"max_new_tokens={capped_tokens}")

    if getattr(sample, "modality", None) == "video" and base_video_max_frames:
        retry_frame_cap = 8 if attempt == 1 else 4
        capped_frames = min(base_video_max_frames, retry_frame_cap)
        if capped_frames < base_video_max_frames:
            try:
                wrapper.video_max_frames = capped_frames
                changes.append(f"video_max_frames={capped_frames}")
            except Exception:
                pass

    if attempt >= 2:
        wrapper.generation_overrides["use_cache"] = False
        changes.append("use_cache=False")

    return ", ".join(changes) if changes else None


def _is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def _looks_like_prompt_echo(sample, answer: str) -> bool:
    if not sample.prompt or not answer:
        return False
    text = answer.strip()
    if not text:
        return False
    match = PROMPT_ECHO_RE.match(text)
    if not match:
        return False

    canon_prompt = clean_prompt_for_echo(sample.prompt)
    if not canon_prompt:
        return False

    canon_user_block = canonicalize_text_block(match.group("user_block"))
    if canon_prompt != canon_user_block:
        return False

    remainder = text[match.end():].strip()
    if not remainder:
        return True

    remainder_compact = re.sub(r"\s+", "", remainder).lower()
    return remainder_compact in {"dk"}


def _needs_seed_retry(model_id: str, sample, answer: str) -> Optional[str]:
    if not (answer or "").strip():
        return "Empty response detected"
    if _is_refusal(answer):
        return "Refusal detected"
    if _looks_like_prompt_echo(sample, answer):
        return "Prompt echo detected"
    if "internvl" in model_id.lower() and (answer or "").strip().lower() == "r":
        return "InternVL placeholder response 'r' detected"
    return None


def _parse_typeb_label(text: str) -> Optional[str]:
    if not text:
        return None
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        lower = line.lower()
        if "likely authentic" in lower:
            return "real"
        if "likely manipulated" in lower:
            return "fake"
        break
    return None


def _parse_mc_choices(text: str) -> Optional[list[str]]:
    if not text:
        return None
    head = " ".join(text.strip().splitlines()[:3])
    letters = sorted(set(re.findall(r"\b([A-E])\b", head.upper())))
    return letters if letters else None


def _parse_tf_answer(text: str) -> Optional[str]:
    if not text:
        return None
    head = " ".join(text.strip().splitlines()[:2]).lower()
    match = re.search(r"\b(yes|no)\b", head)
    if match:
        return match.group(1)
    return None


def _output_path(output_root: Path, task_type: str, split: str, model_id: str, source_path: Path) -> Path:
    return output_root / task_type / sanitize_model_id(model_id) / split / f"{source_path.stem}.jsonl"


def _build_record(
    *,
    task_type: str,
    split: str,
    model_id: str,
    resolved_model_id: str,
    source_path: Path,
    record_id: str,
    sample_payload: dict,
    metadata: dict,
    response: str,
    latency_ms: float,
    fallback_count: int,
    final_seed: int,
    system_hint: Optional[str],
    error: Optional[str],
) -> dict:
    task_meta = task_info(task_type)
    record = {
        "task_type": task_type,
        "task_group": task_meta["task_group"],
        "official_task_name": task_meta["official_task_name"],
        "official_task_id": task_meta["official_task_id"],
        "task_definition": task_meta["task_definition"],
        "evaluation_scope": task_meta["evaluation_scope"],
        "split": split,
        "record_id": record_id,
        "model_id": model_id,
        "response": response,
        "latency_ms": latency_ms,
        "fallback_count": fallback_count,
        "final_seed": final_seed,
        "system_hint": system_hint,
        "source_file": str(source_path),
        "sample": sample_payload,
    }
    if resolved_model_id != model_id:
        record["resolved_model_path"] = resolved_model_id
    record.update(metadata)
    if task_type == "typeb_oeq":
        record["parsed_label"] = _parse_typeb_label(response)
    elif task_type == "mcq":
        record["parsed_choices"] = _parse_mc_choices(response)
    elif task_type == "tfq":
        record["parsed_answer"] = _parse_tf_answer(response)
    if error:
        record["error"] = error
    return record


def _plan_batches(batches: Sequence[BlindBatch], *, task_type: str, split: str, output_root: Path, model_id: str) -> None:
    total = sum(len(batch.items) for batch in batches)
    print(f"[PLAN] task={task_type} split={split} batches={len(batches)} samples={total}")
    for batch in batches:
        output_path = _output_path(output_root, task_type, split, model_id, batch.source_path)
        print(f"  - {batch.source_path.name}: {len(batch.items)} -> {output_path}")


def _ensure_runtime(args: argparse.Namespace, modalities: Sequence[str]) -> None:
    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.cache_dir)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.dry_run:
        return
    if "video" in modalities:
        require_video_support()


def _select_wrapper_cls(model_id: str):
    lower = str(model_id).lower()
    if "qwen3-vl" in lower or "qwen-vl" in lower:
        return Qwen3VLWrapper
    if "phi-4" in lower or "phi4" in lower:
        return Phi4MultimodalWrapper
    raise ValueError(
        "This starter kit only supports Qwen/Qwen3-VL-8B-Instruct and "
        "microsoft/Phi-4-multimodal-instruct. "
        f"Unable to infer a wrapper for '{model_id}'."
    )


def _supported_modalities_for_model(model_id: str) -> list[str]:
    wrapper_cls = _select_wrapper_cls(model_id)
    return list(wrapper_cls.capabilities.supported_modalities)


def _build_wrapper(args: argparse.Namespace, model_id: str):
    load_kwargs = build_loading_kwargs(
        device_map=args.device_map,
        gpus=args.gpus,
        per_gpu_max_memory_gib=args.per_gpu_max_memory_gib,
        flash_attn=bool(args.flash_attn),
        cache_dir=args.cache_dir,
        offline=bool(args.offline),
    )
    wrapper_cls = _select_wrapper_cls(model_id)
    return wrapper_cls(
        model_id=model_id,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=resolve_torch_dtype(args.dtype),
        load_kwargs=load_kwargs,
        video_max_frames=args.video_max_frames,
    )


def _run_batches(
    *,
    args: argparse.Namespace,
    task_type: str,
    split: str,
    batches: Sequence[BlindBatch],
) -> None:
    if not batches:
        print(f"[INFO] No batches found for task={task_type} split={split}.")
        return

    resolved_model_id = maybe_localize(args.model, args.cache_dir) if args.cache_dir else args.model
    wrapper = _build_wrapper(args, resolved_model_id)
    system_hint = task_system_hint(task_type)

    for batch in batches:
        output_path = _output_path(args.output_root, task_type, split, args.model, batch.source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and args.overwrite:
            output_path.unlink()

        seen_ids = _existing_ids(output_path)
        if seen_ids is None:
            output_path.unlink(missing_ok=True)
            seen_ids = set()

        total = len(batch.items)
        done = len(seen_ids)
        if done >= total:
            print(f"[INFO] Skipping {output_path.name}: already complete ({done}/{total}).")
            continue

        mode = "a" if done else "w"
        print(f"[INFO] {task_type}:{split}:{batch.source_path.name} -> {done}/{total} already done.")
        with output_path.open(mode, encoding="utf-8") as handle:
            for index, item in enumerate(batch.items, start=1):
                if item.record_id in seen_ids:
                    continue
                attempt = 0
                oom_retry = False
                response = ""
                error_msg: Optional[str] = None
                latency_ms = 0.0
                base_video_max_frames = getattr(wrapper, "video_max_frames", None)
                while True:
                    wrapper.system_hint = system_hint
                    wrapper.user_prefix = None
                    wrapper.generation_overrides = {}
                    if base_video_max_frames is not None:
                        try:
                            wrapper.video_max_frames = base_video_max_frames
                        except Exception:
                            pass
                    if hasattr(wrapper, "last_usage"):
                        wrapper.last_usage = None
                    retry_config = None
                    if oom_retry:
                        retry_config = _apply_memory_retry_overrides(
                            wrapper=wrapper,
                            sample=item.sample,
                            attempt=attempt,
                            base_max_new_tokens=args.max_new_tokens,
                            base_video_max_frames=base_video_max_frames,
                        )
                    _set_all_seeds(args.seed + attempt)
                    start_ts = time.perf_counter()
                    try:
                        response = wrapper.generate(item.sample)
                        error_msg = None
                    except Exception as exc:  # noqa: BLE001
                        response = ""
                        error_msg = _format_exception_chain(exc)
                    finally:
                        _cleanup_cuda_memory()
                    latency_ms = (time.perf_counter() - start_ts) * 1000.0

                    retry_reason = _needs_seed_retry(args.model, item.sample, response)
                    is_oom = _is_oom_error(error_msg)
                    if is_oom and attempt < args.max_retries:
                        attempt += 1
                        oom_retry = True
                        retry_config_msg = f" with {retry_config}" if retry_config else ""
                        print(f"[INFO] OOM detected -> retry (attempt={attempt}){retry_config_msg}")
                        continue
                    if error_msg and attempt < args.max_retries:
                        retry_reason = retry_reason or "Generation error"
                    if retry_reason and attempt < args.max_retries:
                        attempt += 1
                        oom_retry = False
                        print(f"[INFO] {retry_reason} -> retry (attempt={attempt})")
                        continue
                    break

                record = _build_record(
                    task_type=task_type,
                    split=split,
                    model_id=args.model,
                    resolved_model_id=resolved_model_id,
                    source_path=batch.source_path,
                    record_id=item.record_id,
                    sample_payload=item.sample.to_json(),
                    metadata=item.metadata,
                    response=response,
                    latency_ms=latency_ms,
                    fallback_count=attempt,
                    final_seed=args.seed + attempt,
                    system_hint=system_hint,
                    error=error_msg,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                done += 1
                print(
                    f"[INFO] {task_type}:{split}:{batch.source_path.stem} {done}/{total} "
                    f"({done / total * 100:.2f}%)",
                    end="\r",
                )
        print()
        print(f"[INFO] Saved {output_path}")


def main() -> None:
    args = parse_args()
    args.model = resolve_model_alias(args.model)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False

    requested_modalities = normalize_modalities(args.modalities)
    resolved_model_id = maybe_localize(args.model, args.cache_dir) if args.cache_dir else args.model
    supported_modalities = _supported_modalities_for_model(resolved_model_id)
    modalities = [modality for modality in requested_modalities if modality in supported_modalities]
    skipped_modalities = [modality for modality in requested_modalities if modality not in supported_modalities]
    if skipped_modalities:
        print(
            "[INFO] "
            f"{sanitize_model_id(args.model)} does not support modalities {sorted(skipped_modalities)}. "
            f"Continuing with {modalities}."
        )
    if any(modality == "video" for modality in modalities) and not (
        os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    ):
        print(
            "[INFO] Video runs with variable frame sizes can fragment the CUDA allocator. "
            "Set PYTORCH_ALLOC_CONF=expandable_segments:True before launching Python for better stability."
        )
    if not modalities:
        raise ValueError(
            f"No compatible modalities remain for model '{args.model}'. "
            f"Requested={requested_modalities}, supported={supported_modalities}."
        )
    _ensure_runtime(args, modalities)

    for task_type in expand_tasks(args.task):
        batches = load_batches(
            data_root=args.data_root.resolve(),
            task_type=task_type,
            split=args.split,
            allowed_modalities=modalities,
            max_samples=args.max_samples,
            question_files=args.question_files,
        )
        _plan_batches(
            batches,
            task_type=task_type,
            split=args.split,
            output_root=args.output_root.resolve(),
            model_id=args.model,
        )
        if args.dry_run:
            continue
        _run_batches(args=args, task_type=task_type, split=args.split, batches=batches)


if __name__ == "__main__":
    main()
