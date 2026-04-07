from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from starter_kit._bootstrap import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_OUTPUT_ROOT,
)
from starter_kit.blind_data import expand_tasks, task_info
from starter_kit.oeq_artifact_evaluator import (
    build_mapping_output_dir,
    get_oeq_artifact_evaluator_spec,
    resolve_evaluator_model,
    run_llm_mapping,
)
from starter_kit.runtime.artifact_parser import parse_artifact_map


MC_LETTER_RE = re.compile(r"\b([A-E])\b", flags=re.IGNORECASE)
TF_RE = re.compile(r"\b(true|false)\b", flags=re.IGNORECASE)
OEQ_BASE_COLUMNS = {"sample_id", "media_path", "modality", "track_id", "label"}
QUESTION_MODALITY_MAP = {
    "img": "image",
    "image": "image",
    "vid": "video",
    "video": "video",
    "aud": "audio",
    "audio": "audio",
}
OEQ_EVALUATOR_TASKS = frozenset({"typea_oeq", "typeb_oeq"})
PREDICTION_TASK_DIR_ALIASES = {
    "typeb_oeq": ("typeb_oeq", "detection_oeq"),
    "typea_oeq": ("typea_oeq", "perception_oeq"),
}
SUMMARY_RESULTS_DIRNAME = "evaluation_results"
OEQ_MAPPING_RESULTS_DIRNAME = "oeq_mappings"
SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
TCS_WEIGHTS = {
    "det": 0.4,
    "hal": 0.3,
    "perc": 0.3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate starter-kit predictions against public blind answers.")
    parser.add_argument(
        "--task",
        default="all",
        help="One of: typeb_oeq (Type-B OEQ), typea_oeq (Type-A OEQ), mcq, tfq, all",
    )
    parser.add_argument("--split", default="public_val", choices=["train", "public_val"])
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=str, default=None, help="Optional model id or model directory name")
    parser.add_argument(
        "--typea-export-root",
        "--perception-export-root",
        dest="typea_export_root",
        type=Path,
        default=None,
        help="Optional directory to export OEQ artifact CSVs before scoring.",
    )
    parser.add_argument(
        "--oeq-evaluator-root",
        "--typea-evaluator-root",
        "--perception-evaluator-root",
        dest="oeq_evaluator_root",
        type=Path,
        default=None,
        help=(
            "Optional root for OEQ artifact-mapping outputs. When omitted, mappings are written under "
            "<predictions-root>/evaluation_results/oeq_mappings/."
        ),
    )
    parser.add_argument(
        "--oeq-evaluator-backend",
        "--typea-evaluator-backend",
        "--perception-evaluator-backend",
        dest="oeq_evaluator_backend",
        type=str,
        default="local",
        choices=("local", "openai", "gemini"),
        help="Backend used by the OEQ artifact evaluator.",
    )
    parser.add_argument(
        "--oeq-evaluator-model",
        "--typea-evaluator-model",
        "--perception-evaluator-model",
        dest="oeq_evaluator_model",
        type=str,
        default=None,
        help=(
            "Optional evaluator model id. Defaults depend on the selected backend. "
            "Short aliases: openai=gpt-mini, gemini=gemini-flash-lite."
        ),
    )
    parser.add_argument(
        "--oeq-evaluator-analysis-field",
        "--typea-evaluator-analysis-field",
        "--perception-evaluator-analysis-field",
        dest="oeq_evaluator_analysis_field",
        type=str,
        default="response",
        choices=("analysis_text", "response", "auto"),
        help="Field from starter-kit outputs passed into the evaluator prompt.",
    )
    parser.add_argument(
        "--oeq-evaluator-batch-size",
        "--typea-evaluator-batch-size",
        "--perception-evaluator-batch-size",
        dest="oeq_evaluator_batch_size",
        type=int,
        default=10,
        help="Local OEQ evaluator micro-batch size. Increase it if you have enough GPU memory.",
    )
    parser.add_argument(
        "--oeq-evaluator-poll-interval",
        "--typea-evaluator-poll-interval",
        "--perception-evaluator-poll-interval",
        dest="oeq_evaluator_poll_interval",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--oeq-evaluator-max-parallel-batches",
        "--typea-evaluator-max-parallel-batches",
        "--perception-evaluator-max-parallel-batches",
        dest="oeq_evaluator_max_parallel_batches",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--oeq-evaluator-concurrency",
        "--typea-evaluator-concurrency",
        "--perception-evaluator-concurrency",
        dest="oeq_evaluator_concurrency",
        type=int,
        default=1,
        help="Number of concurrent API requests used by the OEQ evaluator. Local backend stays single-threaded.",
    )
    parser.add_argument(
        "--oeq-evaluator-max-new-tokens",
        "--typea-evaluator-max-new-tokens",
        "--perception-evaluator-max-new-tokens",
        dest="oeq_evaluator_max_new_tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--oeq-evaluator-api-key",
        "--typea-evaluator-api-key",
        "--perception-evaluator-api-key",
        dest="oeq_evaluator_api_key",
        type=str,
        default=None,
        help="Optional evaluator API key. Falls back to backend-specific environment variables when omitted.",
    )
    parser.add_argument(
        "--oeq-evaluator-torch-dtype",
        "--typea-evaluator-torch-dtype",
        "--perception-evaluator-torch-dtype",
        dest="oeq_evaluator_torch_dtype",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--oeq-evaluator-device-map",
        "--typea-evaluator-device-map",
        "--perception-evaluator-device-map",
        dest="oeq_evaluator_device_map",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--oeq-evaluator-per-gpu-max-memory-gib",
        "--typea-evaluator-per-gpu-max-memory-gib",
        "--perception-evaluator-per-gpu-max-memory-gib",
        dest="oeq_evaluator_per_gpu_max_memory_gib",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--oeq-evaluator-flash-attn",
        "--typea-evaluator-flash-attn",
        "--perception-evaluator-flash-attn",
        dest="oeq_evaluator_flash_attn",
        action="store_true",
    )
    parser.add_argument(
        "--oeq-evaluator-cache-dir",
        "--typea-evaluator-cache-dir",
        "--perception-evaluator-cache-dir",
        dest="oeq_evaluator_cache_dir",
        type=str,
        default=str(DEFAULT_MODEL_CACHE_DIR),
    )
    parser.add_argument(
        "--oeq-evaluator-offline",
        "--typea-evaluator-offline",
        "--perception-evaluator-offline",
        dest="oeq_evaluator_offline",
        action="store_true",
    )
    parser.add_argument(
        "--oeq-evaluator-completion-window",
        "--typea-evaluator-completion-window",
        "--perception-evaluator-completion-window",
        dest="oeq_evaluator_completion_window",
        type=str,
        default="24h",
    )
    parser.add_argument(
        "--oeq-evaluator-no-batch",
        "--typea-evaluator-no-batch",
        "--perception-evaluator-no-batch",
        dest="oeq_evaluator_no_batch",
        action="store_true",
    )
    parser.add_argument(
        "--oeq-evaluator-overwrite",
        "--typea-evaluator-overwrite",
        "--perception-evaluator-overwrite",
        dest="oeq_evaluator_overwrite",
        action="store_true",
        help="Re-run the LLM evaluator even if cached mapping outputs already exist.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help=(
            "Optional summary output path. When omitted, results are always written under "
            "<predictions-root>/evaluation_results/."
        ),
    )
    return parser.parse_args()


def sanitize_model_id(model_id: str) -> str:
    raw = str(model_id or "").rstrip("/")
    for part in reversed(Path(raw).parts):
        if part.startswith("models--"):
            tokens = [token for token in part.split("--") if token]
            if len(tokens) >= 3:
                return tokens[-1].replace(":", "_")
    base = raw.split("/")[-1]
    return base.replace(":", "_")


def _safe_filename_token(text: str, fallback: str) -> str:
    token = SAFE_FILENAME_RE.sub("-", str(text or "").strip())
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or fallback


def _uses_oeq_evaluator(task_types: Sequence[str]) -> bool:
    return any(str(task_type or "").strip() in OEQ_EVALUATOR_TASKS for task_type in task_types)


def _build_summary_filename(
    *,
    args: argparse.Namespace,
    model_source: str,
    include_evaluator_tags: bool,
) -> str:
    task_tag = _safe_filename_token(args.task, "all")
    split_tag = _safe_filename_token(args.split, "public_val")
    model_tag = _safe_filename_token(model_source, "all_models")
    if not include_evaluator_tags:
        return f"{task_tag}__{split_tag}__{model_tag}.json"

    backend_tag = _safe_filename_token(args.oeq_evaluator_backend, "local")
    evaluator_model = resolve_evaluator_model(args.oeq_evaluator_backend, args.oeq_evaluator_model)
    evaluator_tag = _safe_filename_token(sanitize_model_id(evaluator_model), "evaluator")
    return f"{task_tag}__{split_tag}__{model_tag}__{backend_tag}__{evaluator_tag}.json"


def _default_summary_out_path(args: argparse.Namespace, *, include_evaluator_tags: bool) -> Path:
    predictions_root = args.predictions_root.resolve()
    model_source = sanitize_model_id(args.model) if args.model else "all_models"
    filename = _build_summary_filename(
        args=args,
        model_source=model_source,
        include_evaluator_tags=include_evaluator_tags,
    )
    return predictions_root / SUMMARY_RESULTS_DIRNAME / filename


def _default_oeq_evaluator_root(predictions_root: Path) -> Path:
    return predictions_root / SUMMARY_RESULTS_DIRNAME / OEQ_MAPPING_RESULTS_DIRNAME


def _default_model_summary_out_path(
    args: argparse.Namespace,
    *,
    model_dir: str,
    include_evaluator_tags: bool,
) -> Path:
    predictions_root = args.predictions_root.resolve()
    filename = _build_summary_filename(
        args=args,
        model_source=model_dir,
        include_evaluator_tags=include_evaluator_tags,
    )
    return predictions_root / SUMMARY_RESULTS_DIRNAME / filename


def _resolve_model_summary_out_path(
    *,
    args: argparse.Namespace,
    model_dir: str,
    model_count: int,
    task_types: Sequence[str],
) -> Path:
    if args.summary_out is None:
        return _default_model_summary_out_path(
            args,
            model_dir=model_dir,
            include_evaluator_tags=_uses_oeq_evaluator(task_types),
        )

    summary_out = args.summary_out.resolve()
    if model_count <= 1:
        return summary_out

    suffix = summary_out.suffix or ".json"
    stem = summary_out.stem if summary_out.suffix else summary_out.name
    model_tag = _safe_filename_token(model_dir, "model")
    return summary_out.parent / f"{stem}__{model_tag}{suffix}"


def _log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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


def _select_metric_fields(task_type: str) -> tuple[str, ...]:
    if task_type == "typea_oeq":
        return ("cover", "chair", "hal_rate", "f_0_5")
    if task_type == "typeb_oeq":
        return ("acc_det",)
    if task_type == "mcq":
        return ("score_mcq",)
    if task_type == "tfq":
        return ("acc_tfq",)
    return tuple()


def _compact_modality_rows(rows: Sequence[dict], metric_fields: Sequence[str]) -> List[dict]:
    compacted: List[dict] = []
    for row in rows or []:
        modality = row.get("modality")
        if modality is None:
            continue
        compact_row = {"modality": modality}
        for field in metric_fields:
            if field in row:
                compact_row[field] = row[field]
        compacted.append(compact_row)
    return compacted


def _compact_result_for_output(result: dict) -> dict:
    task_type = str(result.get("task_type") or "")
    compact = {
        "task_type": task_type,
        "task_group": result.get("task_group"),
        "official_task_name": result.get("official_task_name"),
        "official_task_id": result.get("official_task_id"),
        "task_definition": result.get("task_definition"),
        "evaluation_scope": result.get("evaluation_scope"),
        "predictions_dir": result.get("predictions_dir"),
    }
    if "model_dir" in result:
        compact["model_dir"] = result.get("model_dir")
    if "oeq_mapping_dir" in result:
        compact["oeq_mapping_dir"] = result.get("oeq_mapping_dir")
    elif f"{task_type}_mapping_dir" in result:
        compact["oeq_mapping_dir"] = result.get(f"{task_type}_mapping_dir")
    if "oeq_evaluator_backend" in result:
        compact["oeq_evaluator_backend"] = result.get("oeq_evaluator_backend")
    if "oeq_evaluator_model" in result:
        compact["oeq_evaluator_model"] = result.get("oeq_evaluator_model")
    if "oeq_evaluator_analysis_field" in result:
        compact["oeq_evaluator_analysis_field"] = result.get("oeq_evaluator_analysis_field")

    compact["scores_by_modality"] = _compact_modality_rows(
        result.get("by_modality") or [],
        _select_metric_fields(task_type),
    )
    if task_type == "typeb_oeq" and "artifact_by_modality" in result:
        compact["artifact_scores_by_modality"] = _compact_modality_rows(
            result.get("artifact_by_modality") or [],
            ("cover", "chair", "hal_rate", "f_0_5"),
        )
    return compact


def _metric_index(rows: Sequence[dict], metric_name: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for row in rows or []:
        modality = str(row.get("modality") or "").strip().lower()
        if not modality or metric_name not in row:
            continue
        try:
            metrics[modality] = float(row[metric_name])
        except (TypeError, ValueError):
            continue
    return metrics


def compute_tcs_from_tasks(tasks: Sequence[dict]) -> List[dict]:
    task_map = {
        str(task.get("task_type") or "").strip(): task
        for task in (tasks or [])
        if isinstance(task, dict) and str(task.get("task_type") or "").strip()
    }

    det_scores = _metric_index(task_map.get("typeb_oeq", {}).get("scores_by_modality") or [], "acc_det")
    tfq_scores = _metric_index(task_map.get("tfq", {}).get("scores_by_modality") or [], "acc_tfq")
    mcq_scores = _metric_index(task_map.get("mcq", {}).get("scores_by_modality") or [], "score_mcq")
    typea_hal_scores = _metric_index(task_map.get("typea_oeq", {}).get("scores_by_modality") or [], "f_0_5")
    typeb_hal_scores = _metric_index(
        task_map.get("typeb_oeq", {}).get("artifact_scores_by_modality") or [],
        "f_0_5",
    )

    modalities = sorted(
        set(det_scores)
        | set(tfq_scores)
        | set(mcq_scores)
        | set(typea_hal_scores)
        | set(typeb_hal_scores)
    )
    tcs_rows: List[dict] = []
    for modality in modalities:
        acc_det = det_scores.get(modality)
        acc_tfq = tfq_scores.get(modality)
        score_mcq = mcq_scores.get(modality)
        typea_f_0_5 = typea_hal_scores.get(modality)
        typeb_f_0_5 = typeb_hal_scores.get(modality)
        f_0_5 = (
            (typea_f_0_5 + typeb_f_0_5) / 2.0
            if typea_f_0_5 is not None and typeb_f_0_5 is not None
            else None
        )

        s_det = 100.0 * acc_det if acc_det is not None else None
        s_perc = (
            100.0 * ((0.5 * acc_tfq) + (0.5 * score_mcq))
            if acc_tfq is not None and score_mcq is not None
            else None
        )
        s_hal = 100.0 * f_0_5 if f_0_5 is not None else None
        tcs = (
            (TCS_WEIGHTS["det"] * s_det)
            + (TCS_WEIGHTS["hal"] * s_hal)
            + (TCS_WEIGHTS["perc"] * s_perc)
            if s_det is not None and s_hal is not None and s_perc is not None
            else None
        )

        row = {
            "modality": modality,
            "acc_det": acc_det,
            "acc_tfq": acc_tfq,
            "score_mcq": score_mcq,
            "typea_f_0_5": typea_f_0_5,
            "typeb_f_0_5": typeb_f_0_5,
            "f_0_5": f_0_5,
            "s_det": s_det,
            "s_perc": s_perc,
            "s_hal": s_hal,
            "tcs": tcs,
            "complete": tcs is not None,
        }
        tcs_rows.append(row)
    return tcs_rows


def _model_dir_candidates(model: Optional[str]) -> Optional[set[str]]:
    if not model:
        return None
    raw = str(model).strip()
    return {raw, sanitize_model_id(raw)}


def _read_json_records(path: Path) -> Iterator[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
        return
    if path.suffix != ".json":
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def iter_prediction_records(root: Path) -> Iterator[dict]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        yield from _read_json_records(path)


def iter_mapping_records(root: Path) -> Iterator[dict]:
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _discover_prediction_dirs(
    *,
    predictions_root: Path,
    task_type: str,
    split: str,
    model_filter: Optional[str],
) -> List[tuple[str, Path]]:
    candidates = _model_dir_candidates(model_filter)
    matches: List[tuple[str, Path]] = []
    seen_split_dirs: set[Path] = set()
    task_dir_names = PREDICTION_TASK_DIR_ALIASES.get(task_type, (task_type,))
    for task_dir_name in task_dir_names:
        task_root = predictions_root / task_dir_name
        if not task_root.exists():
            continue
        for model_dir in sorted(task_root.iterdir()):
            if not model_dir.is_dir():
                continue
            if candidates and model_dir.name not in candidates:
                continue
            split_dir = model_dir / split
            if not split_dir.is_dir():
                continue
            resolved_split_dir = split_dir.resolve()
            if resolved_split_dir in seen_split_dirs:
                continue
            seen_split_dirs.add(resolved_split_dir)
            matches.append((model_dir.name, split_dir))
    return matches


def _extract_sample_id(record: dict) -> str:
    for key in ("sample_id", "record_id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    sample = record.get("sample")
    if isinstance(sample, dict):
        value = str(sample.get("sample_id") or "").strip()
        if value:
            return value
    return ""


def _extract_question_id(record: dict) -> str:
    value = str(record.get("question_id") or "").strip()
    if value:
        return value
    sample = record.get("sample")
    if isinstance(sample, dict):
        value = str(sample.get("question_id") or sample.get("sample_id") or "").strip()
        if value:
            return value
    return ""


def _extract_modality(record: dict) -> str:
    value = str(record.get("modality") or "").strip().lower()
    if value:
        return value
    sample = record.get("sample")
    if isinstance(sample, dict):
        value = str(sample.get("modality") or "").strip().lower()
        if value:
            return value
    media_path = str(record.get("media_path") or "").lower()
    if "/audio/" in media_path:
        return "audio"
    if "/video/" in media_path:
        return "video"
    if "/image/" in media_path:
        return "image"
    return "unknown"


def _parse_typeb_label(text: object) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
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


def _normalize_mc_answer(value: object) -> Optional[List[str]]:
    if isinstance(value, list):
        letters = sorted({str(item).strip().upper() for item in value if str(item).strip().upper() in {"A", "B", "C", "D", "E"}})
        return letters if letters else []
    if isinstance(value, str):
        head = " ".join(value.strip().splitlines()[:3]).upper()
        letters = sorted(set(MC_LETTER_RE.findall(head)))
        return letters if letters else []
    return None


def _normalize_question_modality(value: object) -> str:
    return QUESTION_MODALITY_MAP.get(str(value or "").strip().lower(), "unknown")


def _infer_mcq_total_options(question_row: dict) -> Optional[int]:
    question = str(question_row.get("question") or "")
    letters = sorted(set(re.findall(r"(?m)^\s*([A-E])\.", question)))
    if letters:
        return len(letters)
    options = question_row.get("options")
    if isinstance(options, list) and options:
        total = len(options)
        if "none of the options are correct" in question.lower():
            total += 1
        return total
    return None


def _score_mcq_selection(selected: Sequence[str], correct: Sequence[str], total_options: int) -> float:
    selected_set = {str(item).strip().upper() for item in selected if str(item).strip()}
    correct_set = {str(item).strip().upper() for item in correct if str(item).strip()}
    if not correct_set or total_options <= 0:
        return 0.0
    k = len(correct_set)
    wrong_option_count = max(total_options - k, 1)
    score = 0.0
    score += sum(1.0 / k for option in selected_set if option in correct_set)
    score -= sum(1.0 / wrong_option_count for option in selected_set if option not in correct_set)
    return max(0.0, score)


def _normalize_tf_answer(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    match = TF_RE.search(" ".join(value.strip().splitlines()[:2]))
    if not match:
        return None
    return match.group(1).lower() == "true"


def _parse_bool_truth(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"false", "0", "none", "null"}:
        return False
    if text in {"[]", "[ ]"}:
        return False
    return True


def _load_oeq_answers(data_root: Path, split: str) -> Dict[str, dict]:
    answers: Dict[str, dict] = {}
    split_root = data_root / "OEQ" / split
    for modality in ("audio", "image", "video"):
        path = split_root / f"answers_{modality}.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            artifact_names = [name for name in (reader.fieldnames or []) if name not in OEQ_BASE_COLUMNS]
            for row in reader:
                sample_id = str(row.get("sample_id") or "").strip()
                if not sample_id:
                    continue
                ground_truth_artifacts = {
                    artifact: _parse_bool_truth(row.get(artifact))
                    for artifact in artifact_names
                }
                answers[sample_id] = {
                    "label": str(row.get("label") or "").strip().lower(),
                    "modality": modality,
                    "track_id": row.get("track_id"),
                    "media_path": str(row.get("media_path") or "").strip(),
                    "artifact_names": artifact_names,
                    "ground_truth_artifacts": ground_truth_artifacts,
                }
    return answers


def _load_choice_answers(data_root: Path, task_type: str, split: str) -> Dict[str, dict]:
    package_dir = data_root / ("MCQ" if task_type == "mcq" else "TFQ") / split
    package_root = package_dir / "answers.jsonl"
    answers: Dict[str, dict] = {}
    if not package_root.exists():
        return answers
    with package_root.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or "").strip()
            if question_id:
                answers[question_id] = row
    for json_path in sorted(package_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            question_id = str(row.get("question_id") or "").strip()
            if not question_id or question_id not in answers:
                continue
            answers[question_id].setdefault("modality", _normalize_question_modality(row.get("modality")))
            answers[question_id].setdefault("question", row.get("question"))
            if task_type == "mcq":
                answers[question_id].setdefault("options", row.get("options"))
                total_options = _infer_mcq_total_options(row)
                if total_options is not None:
                    answers[question_id]["total_options"] = total_options
    return answers


def _finalize_grouped_stats(stats: dict, by_modality: dict) -> dict:
    stats["by_modality"] = []
    for modality, values in sorted(by_modality.items()):
        row = {"modality": modality, **values}
        matched = int(values.get("matched_answers", 0))
        expected = int(values.get("expected_answers", matched))
        if "correct" in values:
            row["accuracy"] = (values["correct"] / expected) if expected else 0.0
        if "non_empty_responses" in values:
            row["non_empty_rate"] = (values["non_empty_responses"] / matched) if matched else 0.0
        stats["by_modality"].append(row)
    return stats


def _attach_task_metadata(summary: dict, task_type: str) -> dict:
    meta = task_info(task_type)
    summary["task_type"] = task_type
    summary["task_group"] = meta["task_group"]
    summary["official_task_name"] = meta["official_task_name"]
    summary["official_task_id"] = meta["official_task_id"]
    summary["task_definition"] = meta["task_definition"]
    summary["evaluation_scope"] = meta["evaluation_scope"]
    return summary


def score_typeb_oeq(records: Iterable[dict], answers: Dict[str, dict]) -> dict:
    stats = {
        "metric": "detection_accuracy",
        "metric_symbol": "Acc_Det",
        "official_metric_name": "Detection Accuracy",
        "expected_answers": len(answers),
        "total_predictions": 0,
        "matched_answers": 0,
        "correct": 0,
        "invalid_predictions": 0,
        "missing_answers": 0,
        "duplicate_predictions_skipped": 0,
        "error_records": 0,
    }
    by_modality = defaultdict(
        lambda: {
            "expected_answers": 0,
            "matched_answers": 0,
            "correct": 0,
            "invalid_predictions": 0,
            "error_records": 0,
        }
    )
    for answer in answers.values():
        modality = str(answer.get("modality") or "unknown")
        by_modality[modality]["expected_answers"] += 1
    seen: set[str] = set()

    for record in records:
        stats["total_predictions"] += 1
        if record.get("error"):
            stats["error_records"] += 1
        sample_id = _extract_sample_id(record)
        if not sample_id:
            stats["missing_answers"] += 1
            continue
        if sample_id in seen:
            stats["duplicate_predictions_skipped"] += 1
            continue
        seen.add(sample_id)
        answer = answers.get(sample_id)
        if answer is None:
            stats["missing_answers"] += 1
            continue

        modality = str(answer.get("modality") or _extract_modality(record))
        stats["matched_answers"] += 1
        by_modality[modality]["matched_answers"] += 1
        if record.get("error"):
            by_modality[modality]["error_records"] += 1

        pred = record.get("parsed_label")
        if pred not in {"real", "fake"}:
            pred = _parse_typeb_label(record.get("response"))
        truth = str(answer.get("label") or "").strip().lower()
        if pred not in {"real", "fake"} or truth not in {"real", "fake"}:
            stats["invalid_predictions"] += 1
            by_modality[modality]["invalid_predictions"] += 1
            continue
        if pred == truth:
            stats["correct"] += 1
            by_modality[modality]["correct"] += 1

    expected = stats["expected_answers"]
    stats["accuracy"] = (stats["correct"] / expected) if expected else 0.0
    stats["acc_det"] = stats["accuracy"]
    stats["coverage"] = (stats["matched_answers"] / expected) if expected else 0.0
    stats["accuracy_over_expected"] = (
        stats["correct"] / expected if expected else 0.0
    )
    stats = _finalize_grouped_stats(stats, by_modality)
    for row in stats["by_modality"]:
        row["acc_det"] = row.get("accuracy", 0.0)
    return stats


def score_choice(records: Iterable[dict], answers: Dict[str, dict], task_type: str) -> dict:
    if task_type == "mcq":
        stats = {
            "metric": "mcq_score",
            "metric_symbol": "Score_MCQ",
            "official_metric_name": "MCQ Score",
            "expected_answers": len(answers),
            "total_predictions": 0,
            "matched_answers": 0,
            "invalid_predictions": 0,
            "missing_answers": 0,
            "duplicate_predictions_skipped": 0,
            "error_records": 0,
            "score_sum": 0.0,
            "mcq_score": 0.0,
            "score_mcq": 0.0,
            "prediction_coverage": 0.0,
            "score_over_expected": 0.0,
        }
        by_modality = defaultdict(
            lambda: {
                "expected_answers": 0,
                "matched_answers": 0,
                "invalid_predictions": 0,
                "error_records": 0,
                "score_sum": 0.0,
            }
        )
    else:
        stats = {
            "metric": "tfq_accuracy",
            "metric_symbol": "Acc_TFQ",
            "official_metric_name": "TFQ Accuracy",
            "expected_answers": len(answers),
            "total_predictions": 0,
            "matched_answers": 0,
            "correct": 0,
            "invalid_predictions": 0,
            "missing_answers": 0,
            "duplicate_predictions_skipped": 0,
            "error_records": 0,
        }
        by_modality = defaultdict(
            lambda: {
                "expected_answers": 0,
                "matched_answers": 0,
                "correct": 0,
                "invalid_predictions": 0,
                "error_records": 0,
            }
        )
    for answer in answers.values():
        modality = str(answer.get("modality") or "unknown")
        by_modality[modality]["expected_answers"] += 1
    seen: set[str] = set()

    for record in records:
        stats["total_predictions"] += 1
        if record.get("error"):
            stats["error_records"] += 1
        question_id = _extract_question_id(record)
        if not question_id:
            stats["missing_answers"] += 1
            continue
        if question_id in seen:
            stats["duplicate_predictions_skipped"] += 1
            continue
        seen.add(question_id)
        answer = answers.get(question_id)
        if answer is None:
            stats["missing_answers"] += 1
            continue

        modality = str(answer.get("modality") or _extract_modality(record))
        stats["matched_answers"] += 1
        by_modality[modality]["matched_answers"] += 1
        if record.get("error"):
            by_modality[modality]["error_records"] += 1

        if task_type == "mcq":
            pred = _normalize_mc_answer(record.get("parsed_choices"))
            if pred is None:
                pred = _normalize_mc_answer(record.get("response"))
            truth = _normalize_mc_answer(answer.get("ground_truth"))
            total_options = answer.get("total_options")
        else:
            pred = _normalize_tf_answer(record.get("parsed_answer"))
            if pred is None:
                pred = _normalize_tf_answer(record.get("response"))
            truth = _normalize_tf_answer(answer.get("ground_truth"))
            total_options = None

        if pred is None or truth is None or (task_type == "mcq" and not isinstance(total_options, int)):
            stats["invalid_predictions"] += 1
            by_modality[modality]["invalid_predictions"] += 1
            continue
        if task_type == "mcq":
            score = _score_mcq_selection(pred, truth, total_options)
            stats["score_sum"] += score
            by_modality[modality]["score_sum"] += score
        elif pred == truth:
            stats["correct"] += 1
            by_modality[modality]["correct"] += 1

    expected = stats["expected_answers"]
    if task_type == "mcq":
        stats["mcq_score"] = (stats["score_sum"] / expected) if expected else 0.0
        stats["score_mcq"] = stats["mcq_score"]
        stats["prediction_coverage"] = (stats["matched_answers"] / expected) if expected else 0.0
        stats["score_over_expected"] = (
            stats["score_sum"] / expected if expected else 0.0
        )
        stats["by_modality"] = []
        for modality, values in sorted(by_modality.items()):
            expected = int(values.get("expected_answers", 0))
            row = {
                "modality": modality,
                "expected_answers": expected,
                "matched_answers": values["matched_answers"],
                "invalid_predictions": values["invalid_predictions"],
                "error_records": values["error_records"],
                "mcq_score": (values["score_sum"] / expected) if expected else 0.0,
                "score_mcq": (values["score_sum"] / expected) if expected else 0.0,
                "prediction_coverage": (values["matched_answers"] / expected) if expected else 0.0,
                "score_over_expected": (values["score_sum"] / expected) if expected else 0.0,
            }
            stats["by_modality"].append(row)
        return stats

    stats["accuracy"] = (stats["correct"] / expected) if expected else 0.0
    stats["acc_tfq"] = stats["accuracy"]
    stats["coverage"] = (stats["matched_answers"] / expected) if expected else 0.0
    stats["accuracy_over_expected"] = (
        stats["correct"] / expected if expected else 0.0
    )
    stats = _finalize_grouped_stats(stats, by_modality)
    for row in stats["by_modality"]:
        row["acc_tfq"] = row.get("accuracy", 0.0)
    return stats


def _write_oeq_artifact_eval_csv(
    export_root: Path,
    task_type: str,
    model_dir: str,
    split: str,
    rows_by_modality: Dict[str, List[dict]],
    headers_by_modality: Dict[str, List[str]],
) -> Dict[str, str]:
    exported: Dict[str, str] = {}
    for modality, rows in sorted(rows_by_modality.items()):
        if not rows:
            continue
        out_dir = export_root / task_type / model_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task_type}_{modality}.csv"
        header = ["question_id", "sample_path", *headers_by_modality[modality]]
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        exported[modality] = str(out_path)
    return exported


def ensure_oeq_mapping(
    *,
    task_type: str,
    predictions_dir: Path,
    prediction_model_dir: str,
    split: str,
    evaluator_root: Path,
    evaluator_backend: str,
    evaluator_model: Optional[str],
    analysis_field: str,
    batch_size: int,
    poll_interval: int,
    max_parallel_batches: int,
    concurrency: int,
    max_new_tokens: int,
    api_key: Optional[str],
    torch_dtype: str,
    device_map: str,
    per_gpu_max_memory_gib: Optional[int],
    flash_attn: bool,
    cache_dir: Optional[str],
    offline: bool,
    completion_window: str,
    no_batch: bool,
    overwrite: bool,
) -> tuple[Path, str]:
    resolved_model = resolve_evaluator_model(evaluator_backend, evaluator_model)
    mapping_dir = build_mapping_output_dir(
        output_root=evaluator_root,
        task_type=task_type,
        prediction_model_dir=prediction_model_dir,
        split=split,
        backend=evaluator_backend,
        evaluator_model=resolved_model,
        analysis_field=analysis_field,
    )
    resolved_model = run_llm_mapping(
        input_root=predictions_dir,
        output_dir=mapping_dir,
        backend=evaluator_backend,
        model_id=resolved_model,
        analysis_field=analysis_field,
        batch_size=batch_size,
        poll_interval=poll_interval,
        max_parallel_batches=max_parallel_batches,
        concurrency=concurrency,
        max_new_tokens=max_new_tokens,
        api_key=api_key,
        torch_dtype=torch_dtype,
        device_map=device_map,
        per_gpu_max_memory_gib=per_gpu_max_memory_gib,
        flash_attn=flash_attn,
        cache_dir=cache_dir,
        offline=offline,
        completion_window=completion_window,
        no_batch=no_batch,
        skip_existing=not overwrite,
    )
    return mapping_dir, resolved_model


def score_oeq_artifacts(
    records: Iterable[dict],
    answers: Dict[str, dict],
    *,
    task_type: str,
    export_root: Optional[Path] = None,
    model_dir: Optional[str] = None,
    split: Optional[str] = None,
) -> dict:
    stats = {
        "metric": "macro_cover/macro_chair/hal_rate/macro_f_beta",
        "metric_symbols": ["Cover", "CHAIR", "F^0.5"],
        "official_metric_names": [
            "Explanatory Coverage",
            "Hallucinated Artifact Rate",
            "Balanced Interpretability Score",
        ],
        "note": "Artifact extraction is produced by the bundled local evaluator in starter_kit/oeq_artifact_evaluator.py.",
        "artifact_task_type": task_type,
        "expected_answers": len(answers),
        "total_predictions": 0,
        "matched_answers": 0,
        "missing_answers": 0,
        "duplicate_predictions_skipped": 0,
        "error_records": 0,
        "processed_samples": 0,
        "expected_fake_samples": 0,
        "expected_manipulated_samples": 0,
        "fake_samples_scored": 0,
        "manipulated_samples_scored": 0,
        "gt_artifacts": 0,
        "predicted_artifacts": 0,
        "matched_artifacts": 0,
        "macro_cover": 0.0,
        "cover": 0.0,
        "micro_cover": 0.0,
        "macro_chair": 0.0,
        "chair": 0.0,
        "hal_rate": 0.0,
        "macro_f_beta": 0.0,
        "f_0_5": 0.0,
        "prediction_coverage": 0.0,
        "fake_sample_coverage": 0.0,
        "manipulated_sample_coverage": 0.0,
    }
    by_modality = defaultdict(
        lambda: {
            "expected_answers": 0,
            "matched_answers": 0,
            "processed_samples": 0,
            "error_records": 0,
            "expected_fake_samples": 0,
            "fake_samples_scored": 0,
            "gt_artifacts": 0,
            "predicted_artifacts": 0,
            "matched_artifacts": 0,
            "macro_cover_sum": 0.0,
            "macro_chair_sum": 0.0,
            "hal_count": 0,
            "macro_f_beta_sum": 0.0,
        }
    )
    expected_answers_by_modality = defaultdict(int)
    expected_fake_by_modality = defaultdict(int)
    for answer in answers.values():
        modality = str(answer.get("modality") or "unknown")
        expected_answers_by_modality[modality] += 1
        by_modality[modality]["expected_answers"] += 1
        if any(bool(flag) for flag in answer.get("ground_truth_artifacts", {}).values()):
            expected_fake_by_modality[modality] += 1
            by_modality[modality]["expected_fake_samples"] += 1
            stats["expected_fake_samples"] += 1
            stats["expected_manipulated_samples"] += 1

    seen: set[str] = set()
    rows_by_modality: Dict[str, List[dict]] = defaultdict(list)
    headers_by_modality: Dict[str, List[str]] = {}

    for record in records:
        stats["total_predictions"] += 1
        if record.get("error"):
            stats["error_records"] += 1
        sample_id = _extract_sample_id(record)
        if not sample_id:
            stats["missing_answers"] += 1
            continue
        if sample_id in seen:
            stats["duplicate_predictions_skipped"] += 1
            continue
        seen.add(sample_id)
        answer = answers.get(sample_id)
        if answer is None:
            stats["missing_answers"] += 1
            continue

        modality = str(answer.get("modality") or _extract_modality(record))
        stats["matched_answers"] += 1
        stats["processed_samples"] += 1
        by_modality[modality]["matched_answers"] += 1
        by_modality[modality]["processed_samples"] += 1
        if record.get("error"):
            by_modality[modality]["error_records"] += 1
        response = str(record.get("response") or "")
        artifact_names = list(answer.get("artifact_names") or [])
        headers_by_modality.setdefault(modality, artifact_names)
        parsed_map = parse_artifact_map(response)
        predicted = {artifact: bool(parsed_map.get(artifact, False)) for artifact in artifact_names}
        ground_truth = dict(answer.get("ground_truth_artifacts") or {})

        rows_by_modality[modality].append(
            {
                "question_id": sample_id,
                "sample_path": str(answer.get("media_path") or ""),
                **{artifact: bool(predicted.get(artifact, False)) for artifact in artifact_names},
            }
        )

        gt_count = sum(1 for artifact in artifact_names if ground_truth.get(artifact, False))
        pred_count = sum(1 for artifact in artifact_names if predicted.get(artifact, False))
        match_count = sum(
            1
            for artifact in artifact_names
            if ground_truth.get(artifact, False) and predicted.get(artifact, False)
        )

        stats["gt_artifacts"] += gt_count
        stats["predicted_artifacts"] += pred_count
        stats["matched_artifacts"] += match_count
        by_modality[modality]["gt_artifacts"] += gt_count
        by_modality[modality]["predicted_artifacts"] += pred_count
        by_modality[modality]["matched_artifacts"] += match_count

        if gt_count > 0:
            stats["fake_samples_scored"] += 1
            by_modality[modality]["fake_samples_scored"] += 1

            cover = match_count / gt_count
            chair = 1.0 if pred_count == 0 else 1.0 - (match_count / pred_count)
            precision = 1.0 - chair
            beta_sq = 0.25
            denominator = (beta_sq * precision) + cover
            f_beta = ((1 + beta_sq) * precision * cover / denominator) if denominator > 0 else 0.0

            stats["macro_cover"] += cover
            stats["macro_chair"] += chair
            stats["hal_rate"] += 1 if chair > 0 else 0
            stats["macro_f_beta"] += f_beta

            by_modality[modality]["macro_cover_sum"] += cover
            by_modality[modality]["macro_chair_sum"] += chair
            by_modality[modality]["hal_count"] += 1 if chair > 0 else 0
            by_modality[modality]["macro_f_beta_sum"] += f_beta

    fake_samples_scored = stats["fake_samples_scored"]
    expected_fake_samples = stats["expected_fake_samples"]
    missing_fake_samples = max(expected_fake_samples - fake_samples_scored, 0)
    stats["macro_cover"] = (stats["macro_cover"] / expected_fake_samples) if expected_fake_samples else 0.0
    stats["micro_cover"] = (stats["matched_artifacts"] / stats["gt_artifacts"]) if stats["gt_artifacts"] else 0.0
    stats["macro_chair"] = (
        (stats["macro_chair"] + missing_fake_samples) / expected_fake_samples
        if expected_fake_samples
        else 0.0
    )
    stats["hal_rate"] = (
        (stats["hal_rate"] + missing_fake_samples) / expected_fake_samples
        if expected_fake_samples
        else 0.0
    )
    stats["macro_f_beta"] = (stats["macro_f_beta"] / expected_fake_samples) if expected_fake_samples else 0.0
    stats["cover"] = stats["macro_cover"]
    stats["chair"] = stats["macro_chair"]
    stats["f_0_5"] = stats["macro_f_beta"]
    stats["manipulated_samples_scored"] = stats["fake_samples_scored"]
    stats["prediction_coverage"] = (
        stats["matched_answers"] / stats["expected_answers"]
        if stats["expected_answers"]
        else 0.0
    )
    stats["fake_sample_coverage"] = (
        stats["fake_samples_scored"] / stats["expected_fake_samples"]
        if stats["expected_fake_samples"]
        else 0.0
    )
    stats["manipulated_sample_coverage"] = stats["fake_sample_coverage"]

    exported_paths: Dict[str, str] = {}
    if export_root is not None and model_dir and split:
        exported_paths = _write_oeq_artifact_eval_csv(
            export_root,
            task_type,
            model_dir,
            split,
            rows_by_modality,
            headers_by_modality,
        )
        stats["artifact_csv_root"] = str((export_root / task_type / model_dir / split).resolve())

    stats["by_modality"] = []
    for modality, values in sorted(by_modality.items()):
        expected_answers = int(values.get("expected_answers", expected_answers_by_modality.get(modality, 0)))
        expected_fake = int(values.get("expected_fake_samples", expected_fake_by_modality.get(modality, 0)))
        fake_scored = int(values["fake_samples_scored"])
        missing_fake = max(expected_fake - fake_scored, 0)
        row = {
            "modality": modality,
            "expected_answers": expected_answers,
            "matched_answers": values["matched_answers"],
            "processed_samples": values["processed_samples"],
            "expected_fake_samples": expected_fake,
            "expected_manipulated_samples": expected_fake,
            "fake_samples_scored": fake_scored,
            "manipulated_samples_scored": fake_scored,
            "gt_artifacts": values["gt_artifacts"],
            "predicted_artifacts": values["predicted_artifacts"],
            "matched_artifacts": values["matched_artifacts"],
            "micro_cover": (
                values["matched_artifacts"] / values["gt_artifacts"]
                if values["gt_artifacts"]
                else 0.0
            ),
            "macro_cover": (values["macro_cover_sum"] / expected_fake) if expected_fake else 0.0,
            "cover": (values["macro_cover_sum"] / expected_fake) if expected_fake else 0.0,
            "macro_chair": (
                (values["macro_chair_sum"] + missing_fake) / expected_fake
                if expected_fake
                else 0.0
            ),
            "chair": (
                (values["macro_chair_sum"] + missing_fake) / expected_fake
                if expected_fake
                else 0.0
            ),
            "hal_rate": (
                (values["hal_count"] + missing_fake) / expected_fake
                if expected_fake
                else 0.0
            ),
            "macro_f_beta": (values["macro_f_beta_sum"] / expected_fake) if expected_fake else 0.0,
            "f_0_5": (values["macro_f_beta_sum"] / expected_fake) if expected_fake else 0.0,
            "prediction_coverage": (
                values["matched_answers"] / expected_answers
                if expected_answers
                else 0.0
            ),
            "fake_sample_coverage": (
                fake_scored / expected_fake
                if expected_fake
                else 0.0
            ),
            "manipulated_sample_coverage": (
                fake_scored / expected_fake
                if expected_fake
                else 0.0
            ),
            "error_records": values["error_records"],
        }
        if modality in exported_paths:
            row["artifact_csv"] = exported_paths[modality]
        stats["by_modality"].append(row)

    return stats


def score_typea_oeq(
    records: Iterable[dict],
    answers: Dict[str, dict],
    *,
    export_root: Optional[Path] = None,
    model_dir: Optional[str] = None,
    split: Optional[str] = None,
) -> dict:
    return score_oeq_artifacts(
        records,
        answers,
        task_type="typea_oeq",
        export_root=export_root,
        model_dir=model_dir,
        split=split,
    )


def _attach_oeq_artifact_metrics(
    summary: dict,
    artifact_summary: dict,
    *,
    task_type: str,
    mapping_dir: Optional[Path],
    evaluator_backend: str,
    evaluator_model: Optional[str],
    analysis_field: str,
    evaluator_spec: dict,
) -> dict:
    summary["artifact_metric"] = artifact_summary.get("metric")
    summary["artifact_metric_symbols"] = artifact_summary.get("metric_symbols")
    summary["artifact_official_metric_names"] = artifact_summary.get("official_metric_names")
    summary["artifact_note"] = artifact_summary.get("note")
    summary["artifact_task_type"] = task_type
    if mapping_dir is not None:
        summary[f"{task_type}_mapping_dir"] = str(mapping_dir)
    summary[f"{task_type}_evaluator_backend"] = evaluator_backend
    summary[f"{task_type}_evaluator_model"] = evaluator_model
    summary[f"{task_type}_evaluator_analysis_field"] = analysis_field
    summary["oeq_evaluator_backend"] = evaluator_backend
    summary["oeq_evaluator_model"] = evaluator_model
    summary["oeq_evaluator_analysis_field"] = analysis_field

    summary["oeq_artifact_evaluator_spec_version"] = evaluator_spec["spec_version"]
    summary["oeq_artifact_evaluator_prompt_id"] = evaluator_spec["prompt_id"]
    summary["oeq_artifact_evaluator_parser_id"] = evaluator_spec["parser_id"]
    summary["oeq_artifact_official_evaluator_backend"] = evaluator_spec["backend"]
    summary["oeq_artifact_official_evaluator_model"] = evaluator_spec["model_id"]

    for key in (
        "expected_answers",
        "total_predictions",
        "matched_answers",
        "missing_answers",
        "duplicate_predictions_skipped",
        "error_records",
        "processed_samples",
    ):
        if key in artifact_summary:
            summary[f"artifact_{key}"] = artifact_summary[key]

    for key in (
        "expected_fake_samples",
        "expected_manipulated_samples",
        "fake_samples_scored",
        "manipulated_samples_scored",
        "gt_artifacts",
        "predicted_artifacts",
        "matched_artifacts",
        "macro_cover",
        "cover",
        "micro_cover",
        "macro_chair",
        "chair",
        "hal_rate",
        "macro_f_beta",
        "f_0_5",
        "prediction_coverage",
        "fake_sample_coverage",
        "manipulated_sample_coverage",
        "artifact_csv_root",
    ):
        if key in artifact_summary:
            summary[key] = artifact_summary[key]

    if "by_modality" in artifact_summary:
        summary["artifact_by_modality"] = artifact_summary["by_modality"]
    return summary


def evaluate_task_dir(
    task_type: str,
    predictions_dir: Path,
    data_root: Path,
    split: str,
    *,
    model_dir: Optional[str] = None,
    typea_export_root: Optional[Path] = None,
    oeq_evaluator_root: Optional[Path] = None,
    oeq_evaluator_backend: str = "local",
    oeq_evaluator_model: Optional[str] = None,
    oeq_evaluator_analysis_field: str = "response",
    oeq_evaluator_batch_size: int = 10,
    oeq_evaluator_poll_interval: int = 30,
    oeq_evaluator_max_parallel_batches: int = 1,
    oeq_evaluator_concurrency: int = 1,
    oeq_evaluator_max_new_tokens: int = 1024,
    oeq_evaluator_api_key: Optional[str] = None,
    oeq_evaluator_torch_dtype: str = "auto",
    oeq_evaluator_device_map: str = "auto",
    oeq_evaluator_per_gpu_max_memory_gib: Optional[int] = None,
    oeq_evaluator_flash_attn: bool = False,
    oeq_evaluator_cache_dir: Optional[str] = None,
    oeq_evaluator_offline: bool = False,
    oeq_evaluator_completion_window: str = "24h",
    oeq_evaluator_no_batch: bool = False,
    oeq_evaluator_overwrite: bool = False,
) -> dict:
    if task_type == "typeb_oeq":
        answers = _load_oeq_answers(data_root, split)
        records = list(iter_prediction_records(predictions_dir))
        summary = score_typeb_oeq(records, answers)
        if oeq_evaluator_root is not None and model_dir:
            evaluator_spec = get_oeq_artifact_evaluator_spec()
            mapping_dir, resolved_evaluator_model = ensure_oeq_mapping(
                task_type=task_type,
                predictions_dir=predictions_dir,
                prediction_model_dir=model_dir,
                split=split,
                evaluator_root=oeq_evaluator_root,
                evaluator_backend=oeq_evaluator_backend,
                evaluator_model=oeq_evaluator_model,
                analysis_field=oeq_evaluator_analysis_field,
                batch_size=oeq_evaluator_batch_size,
                poll_interval=oeq_evaluator_poll_interval,
                max_parallel_batches=oeq_evaluator_max_parallel_batches,
                concurrency=oeq_evaluator_concurrency,
                max_new_tokens=oeq_evaluator_max_new_tokens,
                api_key=oeq_evaluator_api_key,
                torch_dtype=oeq_evaluator_torch_dtype,
                device_map=oeq_evaluator_device_map,
                per_gpu_max_memory_gib=oeq_evaluator_per_gpu_max_memory_gib,
                flash_attn=oeq_evaluator_flash_attn,
                cache_dir=oeq_evaluator_cache_dir,
                offline=oeq_evaluator_offline,
                completion_window=oeq_evaluator_completion_window,
                no_batch=oeq_evaluator_no_batch,
                overwrite=oeq_evaluator_overwrite,
            )
            artifact_summary = score_oeq_artifacts(
                iter_mapping_records(mapping_dir),
                answers,
                task_type=task_type,
                export_root=typea_export_root,
                model_dir=model_dir,
                split=split,
            )
            summary = _attach_oeq_artifact_metrics(
                summary,
                artifact_summary,
                task_type=task_type,
                mapping_dir=mapping_dir,
                evaluator_backend=oeq_evaluator_backend,
                evaluator_model=resolved_evaluator_model or oeq_evaluator_model,
                analysis_field=oeq_evaluator_analysis_field,
                evaluator_spec=evaluator_spec,
            )
    elif task_type == "typea_oeq":
        evaluator_spec = get_oeq_artifact_evaluator_spec()
        if oeq_evaluator_root is None:
            raise ValueError("oeq_evaluator_root is required for typea_oeq evaluation.")
        if not model_dir:
            raise ValueError("model_dir is required for typea_oeq evaluation.")
        mapping_dir, resolved_evaluator_model = ensure_oeq_mapping(
            task_type=task_type,
            predictions_dir=predictions_dir,
            prediction_model_dir=model_dir,
            split=split,
            evaluator_root=oeq_evaluator_root,
            evaluator_backend=oeq_evaluator_backend,
            evaluator_model=oeq_evaluator_model,
            analysis_field=oeq_evaluator_analysis_field,
            batch_size=oeq_evaluator_batch_size,
            poll_interval=oeq_evaluator_poll_interval,
            max_parallel_batches=oeq_evaluator_max_parallel_batches,
            concurrency=oeq_evaluator_concurrency,
            max_new_tokens=oeq_evaluator_max_new_tokens,
            api_key=oeq_evaluator_api_key,
            torch_dtype=oeq_evaluator_torch_dtype,
            device_map=oeq_evaluator_device_map,
            per_gpu_max_memory_gib=oeq_evaluator_per_gpu_max_memory_gib,
            flash_attn=oeq_evaluator_flash_attn,
            cache_dir=oeq_evaluator_cache_dir,
            offline=oeq_evaluator_offline,
            completion_window=oeq_evaluator_completion_window,
            no_batch=oeq_evaluator_no_batch,
            overwrite=oeq_evaluator_overwrite,
        )
        records = list(iter_mapping_records(mapping_dir))
        summary = score_oeq_artifacts(
            records,
            _load_oeq_answers(data_root, split),
            task_type=task_type,
            export_root=typea_export_root,
            model_dir=model_dir,
            split=split,
        )
        summary["typea_mapping_dir"] = str(mapping_dir)
        summary["typea_evaluator_backend"] = oeq_evaluator_backend
        summary["typea_evaluator_model"] = resolved_evaluator_model or oeq_evaluator_model
        summary["typea_evaluator_analysis_field"] = oeq_evaluator_analysis_field
        summary["typea_evaluator_spec_version"] = evaluator_spec["spec_version"]
        summary["typea_evaluator_prompt_id"] = evaluator_spec["prompt_id"]
        summary["typea_evaluator_parser_id"] = evaluator_spec["parser_id"]
        summary["typea_official_evaluator_backend"] = evaluator_spec["backend"]
        summary["typea_official_evaluator_model"] = evaluator_spec["model_id"]
        summary["oeq_mapping_dir"] = str(mapping_dir)
        summary["oeq_evaluator_backend"] = oeq_evaluator_backend
        summary["oeq_evaluator_model"] = resolved_evaluator_model or oeq_evaluator_model
        summary["oeq_evaluator_analysis_field"] = oeq_evaluator_analysis_field
        summary["oeq_artifact_evaluator_spec_version"] = evaluator_spec["spec_version"]
        summary["oeq_artifact_evaluator_prompt_id"] = evaluator_spec["prompt_id"]
        summary["oeq_artifact_evaluator_parser_id"] = evaluator_spec["parser_id"]
        summary["oeq_artifact_official_evaluator_backend"] = evaluator_spec["backend"]
        summary["oeq_artifact_official_evaluator_model"] = evaluator_spec["model_id"]
    else:
        records = list(iter_prediction_records(predictions_dir))
        summary = score_choice(records, _load_choice_answers(data_root, task_type, split), task_type)
    summary = _attach_task_metadata(summary, task_type)
    summary["predictions_dir"] = str(predictions_dir)
    return summary


def main() -> None:
    args = parse_args()
    predictions_root = args.predictions_root.resolve()
    data_root = args.data_root.resolve()
    oeq_evaluator_root = (
        args.oeq_evaluator_root.resolve()
        if args.oeq_evaluator_root
        else _default_oeq_evaluator_root(predictions_root)
    )
    jobs: List[tuple[str, str, Path]] = []
    for task_type in expand_tasks(args.task):
        for model_dir, split_dir in _discover_prediction_dirs(
            predictions_root=predictions_root,
            task_type=task_type,
            split=args.split,
            model_filter=args.model,
        ):
            jobs.append((task_type, model_dir, split_dir))

    if not jobs:
        raise SystemExit("No matching prediction directories found.")

    model_dirs = sorted({model_dir for _, model_dir, _ in jobs})
    results_by_model: Dict[str, List[dict]] = defaultdict(list)
    _write_progress_line("Evaluate", 0, len(jobs))
    for job_index, (task_type, model_dir, split_dir) in enumerate(jobs, start=1):
        result = evaluate_task_dir(
            task_type,
            split_dir,
            data_root,
            args.split,
            model_dir=model_dir,
            typea_export_root=args.typea_export_root.resolve() if args.typea_export_root else None,
            oeq_evaluator_root=oeq_evaluator_root,
            oeq_evaluator_backend=args.oeq_evaluator_backend,
            oeq_evaluator_model=args.oeq_evaluator_model,
            oeq_evaluator_analysis_field=args.oeq_evaluator_analysis_field,
            oeq_evaluator_batch_size=args.oeq_evaluator_batch_size,
            oeq_evaluator_poll_interval=args.oeq_evaluator_poll_interval,
            oeq_evaluator_max_parallel_batches=args.oeq_evaluator_max_parallel_batches,
            oeq_evaluator_concurrency=args.oeq_evaluator_concurrency,
            oeq_evaluator_max_new_tokens=args.oeq_evaluator_max_new_tokens,
            oeq_evaluator_api_key=args.oeq_evaluator_api_key,
            oeq_evaluator_torch_dtype=args.oeq_evaluator_torch_dtype,
            oeq_evaluator_device_map=args.oeq_evaluator_device_map,
            oeq_evaluator_per_gpu_max_memory_gib=args.oeq_evaluator_per_gpu_max_memory_gib,
            oeq_evaluator_flash_attn=args.oeq_evaluator_flash_attn,
            oeq_evaluator_cache_dir=args.oeq_evaluator_cache_dir,
            oeq_evaluator_offline=args.oeq_evaluator_offline,
            oeq_evaluator_completion_window=args.oeq_evaluator_completion_window,
            oeq_evaluator_no_batch=args.oeq_evaluator_no_batch,
            oeq_evaluator_overwrite=args.oeq_evaluator_overwrite,
        )
        result["model_dir"] = model_dir
        results_by_model[model_dir].append(_compact_result_for_output(result))
        _write_progress_line("Evaluate", job_index, len(jobs))

    written_summaries: List[dict] = []
    for model_dir in model_dirs:
        model_task_types = sorted({str(row.get("task_type") or "") for row in results_by_model.get(model_dir, [])})
        uses_oeq_evaluator = _uses_oeq_evaluator(model_task_types)
        summary_out = _resolve_model_summary_out_path(
            args=args,
            model_dir=model_dir,
            model_count=len(model_dirs),
            task_types=model_task_types,
        )
        model_summary = {
            "split": args.split,
            "model_dir": model_dir,
            "predictions_root": str(predictions_root),
            "data_root": str(data_root),
            "summary_out": str(summary_out),
            "tasks": sorted(
                results_by_model.get(model_dir, []),
                key=lambda row: str(row.get("task_type") or ""),
            ),
        }
        if uses_oeq_evaluator:
            model_summary["oeq_evaluator_root"] = str(oeq_evaluator_root)
            model_summary["oeq_evaluator_concurrency"] = args.oeq_evaluator_concurrency
        model_summary["tcs_weights"] = dict(TCS_WEIGHTS)
        model_summary["tcs_formula"] = (
            "tcs = 0.4 * (100 * acc_det) + 0.3 * (100 * ((typea_f_0_5 + typeb_f_0_5) / 2)) "
            "+ 0.3 * (100 * (0.5 * acc_tfq + 0.5 * score_mcq))"
        )
        model_summary["tcs_by_modality"] = compute_tcs_from_tasks(model_summary["tasks"])
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(model_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written_summaries.append(
            {
                "model_dir": model_dir,
                "summary_out": str(summary_out),
            }
        )
        _log_progress(f"[Evaluate] Summary written to {summary_out}")

    print(json.dumps({"written_summaries": written_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
