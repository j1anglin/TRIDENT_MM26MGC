from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any, Dict, Iterator, List, Optional, Sequence

from starter_kit.runtime.artifact_parser import MAPPING_PROMPT
from starter_kit.runtime.torch import build_loading_kwargs, resolve_torch_dtype
from starter_kit.runtime.types import ModelResponse, TaskSample
from starter_kit.runtime.wrappers.gemini_text import GeminiTextWrapper
from starter_kit.runtime.wrappers.openai_text import OpenAITextWrapper
from starter_kit.runtime.wrappers.text import AutoTextWrapper


DEFAULT_EVALUATOR_MODELS = {
    "local": "Qwen/Qwen3-8B",
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.1-flash-lite-preview",
}

EVALUATOR_MODEL_ALIASES = {
    "openai": {
        "gpt-mini": "gpt-5.4-mini",
    },
    "gemini": {
        "gemini-flash-lite": "gemini-3.1-flash-lite-preview",
    },
}

OEQ_ARTIFACT_EVALUATOR_SPEC = {
    "spec_version": "oeq-openai-gpt-5.4-mini-v1",
    "backend": "openai",
    "model_id": DEFAULT_EVALUATOR_MODELS["openai"],
    "prompt_id": "starter_kit.runtime.artifact_parser:MAPPING_PROMPT@v1",
    "parser_id": "starter_kit.runtime.artifact_parser:parse_artifact_map@v1",
}

SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]")


def sanitize_model_id(model_id: str) -> str:
    raw = str(model_id or "").rstrip("/")
    for part in reversed(Path(raw).parts):
        if part.startswith("models--"):
            tokens = [token for token in part.split("--") if token]
            if len(tokens) >= 3:
                return tokens[-1].replace(":", "_")
    base = raw.split("/")[-1]
    return base.replace(":", "_")


def get_oeq_artifact_evaluator_spec() -> Dict[str, str]:
    return dict(OEQ_ARTIFACT_EVALUATOR_SPEC)


def get_typea_evaluator_spec() -> Dict[str, str]:
    return get_oeq_artifact_evaluator_spec()


def _sanitize_token(text: str, fallback: str) -> str:
    token = SAFE_TOKEN_RE.sub("-", str(text or "").strip())
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or fallback


def resolve_evaluator_model(backend: str, model_id: Optional[str]) -> str:
    normalized_backend = str(backend or "local").strip().lower()
    if normalized_backend not in DEFAULT_EVALUATOR_MODELS:
        allowed = ", ".join(sorted(DEFAULT_EVALUATOR_MODELS))
        raise ValueError(f"Unsupported evaluator backend '{backend}'. Allowed: {allowed}.")
    if model_id:
        resolved_model = str(model_id).strip()
        alias_map = EVALUATOR_MODEL_ALIASES.get(normalized_backend, {})
        return alias_map.get(resolved_model.lower(), resolved_model)
    return DEFAULT_EVALUATOR_MODELS[normalized_backend]


def build_mapping_output_dir(
    *,
    output_root: Path,
    task_type: str,
    prediction_model_dir: str,
    split: str,
    backend: str,
    evaluator_model: str,
    analysis_field: str,
) -> Path:
    evaluator_tag = "__".join(
        [
            _sanitize_token(backend, "local"),
            _sanitize_token(sanitize_model_id(evaluator_model), "evaluator"),
            _sanitize_token(analysis_field, "response"),
        ]
    )
    task_tag = _sanitize_token(task_type, "oeq")
    return output_root / task_tag / prediction_model_dir / split / evaluator_tag


def _write_progress_line(label: str, completed: int, total: int) -> None:
    if total <= 0:
        percent = 100.0
        completed = 0
        total = 0
        filled = 24
    else:
        percent = (completed / total) * 100.0
        filled = min(24, int((completed / total) * 24))
    bar = "#" * filled + "-" * (24 - filled)
    sys.stderr.write(f"\r[{label}] [{bar}] {percent:5.1f}% ({completed}/{total})")
    sys.stderr.flush()
    if completed >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _build_wrapper(
    *,
    normalized_backend: str,
    resolved_model: str,
    max_new_tokens: int,
    api_key: Optional[str],
    torch_dtype: str,
    device_map: str,
    per_gpu_max_memory_gib: Optional[int],
    flash_attn: bool,
    cache_dir: Optional[str],
    offline: bool,
):
    if normalized_backend == "local":
        load_kwargs = build_loading_kwargs(
            device_map=device_map,
            gpus=None,
            per_gpu_max_memory_gib=per_gpu_max_memory_gib,
            flash_attn=flash_attn,
            cache_dir=cache_dir,
            offline=offline,
        )
        return AutoTextWrapper(
            resolved_model,
            max_new_tokens=max_new_tokens,
            torch_dtype=resolve_torch_dtype(torch_dtype),
            load_kwargs=load_kwargs,
        )

    if offline:
        raise ValueError(
            f"Evaluator backend '{normalized_backend}' does not support offline mode. "
            "Remove --oeq-evaluator-offline or use --oeq-evaluator-backend local."
        )
    wrapper_cls = OpenAITextWrapper if normalized_backend == "openai" else GeminiTextWrapper
    return wrapper_cls(
        resolved_model,
        max_new_tokens=max_new_tokens,
        api_key=api_key,
    )


def _generate_and_write_mapping_record(
    *,
    sample: TaskSample,
    sample_path: Path,
    wrapper,
    resolved_model: str,
) -> Dict[str, Any]:
    return _generate_mapping_record(
        sample=sample,
        sample_path=sample_path,
        wrapper=wrapper,
        resolved_model=resolved_model,
    )


def _build_mapping_record(
    *,
    sample: TaskSample,
    resolved_model: str,
    response_text: str,
    latency_ms: float,
    usage_metadata: Optional[Dict[str, Any]],
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    record = ModelResponse(
        model_id=resolved_model,
        sample=sample,
        response=response_text,
        latency_ms=latency_ms,
        usage_metadata=usage_metadata,
    ).to_json()
    if error_message:
        record["error"] = error_message
    return record


def _write_mapping_record(sample_path: Path, record: Dict[str, Any]) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _generate_mapping_record(
    *,
    sample: TaskSample,
    sample_path: Path,
    wrapper,
    resolved_model: str,
) -> Dict[str, Any]:
    error_message = None
    response_text = ""
    usage_metadata = None
    start = time.time()
    try:
        if hasattr(wrapper, "last_usage"):
            wrapper.last_usage = None
        response_text = wrapper.generate(sample)
        usage_metadata = getattr(wrapper, "last_usage", None)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        response_text = f"[ERROR] {error_message}"

    record = _build_mapping_record(
        sample=sample,
        resolved_model=resolved_model,
        response_text=response_text,
        latency_ms=(time.time() - start) * 1000.0,
        usage_metadata=usage_metadata,
        error_message=error_message,
    )
    _write_mapping_record(sample_path, record)
    return record


def _generate_mapping_records_batch(
    *,
    samples: Sequence[TaskSample],
    sample_paths: Sequence[Path],
    wrapper,
    resolved_model: str,
) -> List[Dict[str, Any]]:
    if not samples:
        return []

    start = time.time()
    try:
        if hasattr(wrapper, "last_usage"):
            wrapper.last_usage = None
        responses = wrapper.generate_batch(samples)
        usage_metadata = getattr(wrapper, "last_usage", None)
        if len(responses) != len(samples):
            raise RuntimeError(
                f"Batch evaluator returned {len(responses)} responses for {len(samples)} samples."
            )
    except Exception:
        return [
            _generate_mapping_record(
                sample=sample,
                sample_path=sample_path,
                wrapper=wrapper,
                resolved_model=resolved_model,
            )
            for sample, sample_path in zip(samples, sample_paths)
        ]

    latency_ms = (time.time() - start) * 1000.0
    per_sample_latency_ms = latency_ms / len(samples)
    records: List[Dict[str, Any]] = []
    for sample, sample_path, response_text in zip(samples, sample_paths, responses):
        record = _build_mapping_record(
            sample=sample,
            resolved_model=resolved_model,
            response_text="" if response_text is None else str(response_text),
            latency_ms=per_sample_latency_ms,
            usage_metadata=usage_metadata,
        )
        _write_mapping_record(sample_path, record)
        records.append(record)
    return records


def run_llm_mapping(
    *,
    input_root: Path,
    output_dir: Path,
    backend: str,
    model_id: Optional[str] = None,
    analysis_field: str = "response",
    batch_size: int = 1,
    poll_interval: int = 30,
    max_parallel_batches: int = 1,
    concurrency: int = 1,
    max_new_tokens: int = 1024,
    api_key: Optional[str] = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    per_gpu_max_memory_gib: Optional[int] = None,
    flash_attn: bool = False,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    completion_window: str = "24h",
    no_batch: bool = False,
    skip_existing: bool = True,
) -> str:
    normalized_backend = str(backend or "local").strip().lower()
    resolved_model = resolve_evaluator_model(normalized_backend, model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _collect_analysis_samples(
        input_root=input_root,
        output_dir=output_dir,
        analysis_field=analysis_field,
    )
    total_samples = len(samples)
    _write_progress_line("OEQ mapping", 0, total_samples)

    pending: List[tuple[int, TaskSample, Path]] = []
    completed = 0
    for index, sample in enumerate(samples, start=1):
        sample_path = _output_path_for(sample, output_dir)
        if skip_existing and sample_path.exists():
            completed += 1
            _write_progress_line("OEQ mapping", completed, total_samples)
            continue
        pending.append((index, sample, sample_path))

    effective_concurrency = 1 if normalized_backend == "local" else max(1, int(concurrency or 1))
    effective_concurrency = min(effective_concurrency, len(pending)) if pending else effective_concurrency

    if effective_concurrency <= 1:
        wrapper = _build_wrapper(
            normalized_backend=normalized_backend,
            resolved_model=resolved_model,
            max_new_tokens=max_new_tokens,
            api_key=api_key,
            torch_dtype=torch_dtype,
            device_map=device_map,
            per_gpu_max_memory_gib=per_gpu_max_memory_gib,
            flash_attn=flash_attn,
            cache_dir=cache_dir,
            offline=offline,
        )
        local_batch_size = max(1, int(batch_size or 1)) if normalized_backend == "local" else 1
        if normalized_backend == "local" and local_batch_size > 1:
            for start_index in range(0, len(pending), local_batch_size):
                chunk = pending[start_index : start_index + local_batch_size]
                samples_chunk = [sample for _, sample, _ in chunk]
                sample_paths_chunk = [sample_path for _, _, sample_path in chunk]
                records = _generate_mapping_records_batch(
                    samples=samples_chunk,
                    sample_paths=sample_paths_chunk,
                    wrapper=wrapper,
                    resolved_model=resolved_model,
                )
                for _ in records:
                    completed += 1
                    _write_progress_line("OEQ mapping", completed, total_samples)
            return resolved_model

        for _, sample, sample_path in pending:
            _generate_mapping_record(
                sample=sample,
                sample_path=sample_path,
                wrapper=wrapper,
                resolved_model=resolved_model,
            )
            completed += 1
            _write_progress_line("OEQ mapping", completed, total_samples)
        return resolved_model

    thread_state = local()

    def _get_thread_wrapper():
        wrapper = getattr(thread_state, "wrapper", None)
        if wrapper is None:
            wrapper = _build_wrapper(
                normalized_backend=normalized_backend,
                resolved_model=resolved_model,
                max_new_tokens=max_new_tokens,
                api_key=api_key,
                torch_dtype=torch_dtype,
                device_map=device_map,
                per_gpu_max_memory_gib=per_gpu_max_memory_gib,
                flash_attn=flash_attn,
                cache_dir=cache_dir,
                offline=offline,
            )
            thread_state.wrapper = wrapper
        return wrapper

    def _worker(sample: TaskSample, sample_path: Path) -> None:
        _generate_and_write_mapping_record(
            sample=sample,
            sample_path=sample_path,
            wrapper=_get_thread_wrapper(),
            resolved_model=resolved_model,
        )

    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        futures = [
            executor.submit(_worker, sample, sample_path)
            for _, sample, sample_path in pending
        ]
        for future in as_completed(futures):
            future.result()
            completed += 1
            _write_progress_line("OEQ mapping", completed, total_samples)
    return resolved_model


def _is_under_path(path: Path, root: Optional[Path]) -> bool:
    if root is None:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sanitize_task_path(path: Path) -> str:
    parts = [p for p in path.parts if p not in ("", ".")]
    if not parts:
        return "analysis_text"
    cleaned = [_sanitize_token(part, "analysis_text") for part in parts]
    cleaned = [part for part in cleaned if part]
    return "/".join(cleaned) if cleaned else "analysis_text"


def _iter_result_records(result_path: Path) -> Iterator[Dict[str, object]]:
    if result_path.suffix == ".jsonl":
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload
        return

    if result_path.suffix != ".json":
        return

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                yield entry


def _extract_analysis_text(record: Dict[str, object], *, analysis_field: str) -> Optional[str]:
    field = (analysis_field or "response").strip().lower()
    if field == "response":
        response_text = record.get("response")
        if response_text:
            return str(response_text)
        field = "analysis_text"

    if field in {"analysis_text", "auto"}:
        sample = record.get("sample")
        if isinstance(sample, dict):
            direct_text = sample.get("analysis_text")
            if direct_text:
                return str(direct_text)
            media_meta = sample.get("media_meta")
            if isinstance(media_meta, dict):
                meta_text = media_meta.get("analysis_text")
                if meta_text:
                    return str(meta_text)

        direct_text = record.get("analysis_text")
        if direct_text:
            return str(direct_text)

    if field == "auto":
        response_text = record.get("response")
        if response_text:
            return str(response_text)
    return None


def _output_path_for(sample: TaskSample, output_dir: Path) -> Path:
    task_name = sample.task or "eval_hallucination"
    return output_dir / task_name / f"{sample.sample_id}.json"


def _collect_analysis_samples(
    *,
    input_root: Path,
    output_dir: Path,
    analysis_field: str,
) -> List[TaskSample]:
    collected: List[TaskSample] = []
    for result_path in sorted(input_root.rglob("*.json")) + sorted(input_root.rglob("*.jsonl")):
        if _is_under_path(result_path, output_dir):
            continue
        if not result_path.is_file():
            continue

        rel_path = result_path.relative_to(input_root)
        for record_index, record in enumerate(_iter_result_records(result_path)):
            analysis_text = _extract_analysis_text(record, analysis_field=analysis_field)
            if not analysis_text:
                continue
            analysis_text = str(analysis_text).strip()
            if not analysis_text:
                continue

            sample_meta = record.get("sample") if isinstance(record, dict) else None
            raw_sample_id = None
            if isinstance(sample_meta, dict):
                raw_sample_id = sample_meta.get("sample_id")
            if raw_sample_id is None:
                raw_sample_id = rel_path.stem if record_index == 0 else f"{rel_path.stem}-{record_index}"
            sample_id = _sanitize_token(str(raw_sample_id), rel_path.stem or "sample")

            raw_task = None
            if isinstance(sample_meta, dict):
                raw_task = sample_meta.get("task")
            task_name = (
                _sanitize_task_path(Path(str(raw_task)))
                if raw_task
                else _sanitize_task_path(rel_path.parent)
            )

            media_meta = {}
            if isinstance(sample_meta, dict):
                raw_media_meta = sample_meta.get("media_meta")
                if isinstance(raw_media_meta, dict):
                    media_meta.update(raw_media_meta)
                if sample_meta.get("modality"):
                    media_meta.setdefault("source_modality", sample_meta.get("modality"))
                if sample_meta.get("task"):
                    media_meta.setdefault("source_sample_task", sample_meta.get("task"))
                if sample_meta.get("sample_id"):
                    media_meta.setdefault("source_sample_id", sample_meta.get("sample_id"))
            if record.get("model_id"):
                media_meta.setdefault("source_model_id", record.get("model_id"))
            media_meta.setdefault("analysis_text", analysis_text)
            media_meta.setdefault("source_record_path", str(result_path))
            media_meta.setdefault("source_relative_path", rel_path.as_posix())

            collected.append(
                TaskSample(
                    task=task_name,
                    sample_id=sample_id,
                    modality="text",
                    prompt=MAPPING_PROMPT.format(RESPONSE=analysis_text),
                    fake_path=str(result_path),
                    relative_fake_path=str(rel_path),
                    label="analysis",
                    media_meta=media_meta,
                )
            )
    return collected
