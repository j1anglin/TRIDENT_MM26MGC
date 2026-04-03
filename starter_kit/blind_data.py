from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from starter_kit.runtime.prompts import (
    PERCEPTION_MC_SYSTEM_HINT,
    PERCEPTION_TF_SYSTEM_HINT,
    TYPEA_OEQ_PROMPTS,
    TYPEA_SYSTEM_HINT,
    TYPEB_OEQ_PROMPTS,
    TYPEB_SYSTEM_HINT,
)
from starter_kit.runtime.text import strip_modality_tags
from starter_kit.runtime.types import TaskSample


MODALITY_MAP = {
    "img": "image",
    "image": "image",
    "vid": "video",
    "video": "video",
    "aud": "audio",
    "audio": "audio",
}

TASK_ALIASES = {
    "detection": "typeb_oeq",
    "detection_oe": "typeb_oeq",
    "detection_oeq": "typeb_oeq",
    "type_b_oeq": "typeb_oeq",
    "type-b-oeq": "typeb_oeq",
    "typeb_oeq": "typeb_oeq",
    "detection_type_b_oeq": "typeb_oeq",
    "perception": "typea_oeq",
    "perception_oe": "typea_oeq",
    "perception_oeq": "typea_oeq",
    "type_a_oeq": "typea_oeq",
    "type-a-oeq": "typea_oeq",
    "typea_oeq": "typea_oeq",
    "perception_type_a_oeq": "typea_oeq",
    "mc": "mcq",
    "mcq": "mcq",
    "tf": "tfq",
    "tfq": "tfq",
}

TASK_ORDER = ("typeb_oeq", "typea_oeq", "mcq", "tfq")

SYSTEM_HINTS = {
    "typeb_oeq": TYPEB_SYSTEM_HINT,
    "typea_oeq": TYPEA_SYSTEM_HINT,
    "mcq": PERCEPTION_MC_SYSTEM_HINT,
    "tfq": PERCEPTION_TF_SYSTEM_HINT,
}

TASK_METADATA = {
    "typeb_oeq": {
        "task_group": "detection",
        "official_task_name": "Type-B OEQ",
        "official_task_id": "detection_type_b_oeq",
        "task_definition": "Holistic authentic-vs-manipulated detection on both authentic and manipulated samples.",
        "evaluation_scope": "both authentic and manipulated samples",
    },
    "typea_oeq": {
        "task_group": "perception",
        "official_task_name": "Type-A OEQ",
        "official_task_id": "perception_type_a_oeq",
        "task_definition": "Open-ended artifact localization and identification, scored on manipulated samples only.",
        "evaluation_scope": "manipulated samples only",
    },
    "mcq": {
        "task_group": "perception",
        "official_task_name": "MCQ",
        "official_task_id": "perception_mcq",
        "task_definition": "Structured multi-label artifact identification, scored on manipulated samples only.",
        "evaluation_scope": "manipulated samples only",
    },
    "tfq": {
        "task_group": "perception",
        "official_task_name": "TFQ",
        "official_task_id": "perception_tfq",
        "task_definition": "Binary artifact verification, scored on manipulated samples only.",
        "evaluation_scope": "manipulated samples only",
    },
}

OEQ_PROMPTS = {
    "typeb_oeq": dict(TYPEB_OEQ_PROMPTS),
    "typea_oeq": dict(TYPEA_OEQ_PROMPTS),
}


@dataclass
class BlindItem:
    sample: TaskSample
    record_id: str
    metadata: dict


@dataclass
class BlindBatch:
    task_type: str
    split: str
    source_path: Path
    items: List[BlindItem]


def normalize_task_name(task_name: str) -> str:
    key = str(task_name or "").strip().lower()
    if key == "all":
        return "all"
    task = TASK_ALIASES.get(key)
    if not task:
        allowed = ", ".join(sorted({"all", *TASK_ALIASES.keys()}))
        raise ValueError(f"Unsupported task '{task_name}'. Allowed: {allowed}")
    return task


def normalize_modalities(modalities: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for raw in modalities or []:
        modality = MODALITY_MAP.get(str(raw).strip().lower())
        if modality and modality not in normalized:
            normalized.append(modality)
    return normalized or ["image", "video", "audio"]


def task_system_hint(task_type: str) -> Optional[str]:
    return SYSTEM_HINTS[task_type]


def task_info(task_type: str) -> dict:
    return dict(TASK_METADATA[task_type])


def expand_tasks(task_name: str) -> List[str]:
    normalized = normalize_task_name(task_name)
    if normalized == "all":
        return list(TASK_ORDER)
    return [normalized]


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_truth_from_row(row: dict) -> Dict[str, bool]:
    truth: Dict[str, bool] = {}
    for key, value in row.items():
        if key in {"sample_id", "media_path", "modality", "track_id", "label"}:
            continue
        truth[key] = str(value).strip().lower() == "true"
    return truth


def _load_oeq_answers(split_root: Path) -> Dict[str, dict]:
    answers_by_sample: Dict[str, dict] = {}
    for modality in ("audio", "image", "video"):
        path = split_root / f"answers_{modality}.csv"
        if not path.exists():
            continue
        for row in _read_csv_rows(path):
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                continue
            answers_by_sample[sample_id] = {
                "label": row.get("label"),
                "ground_truth_artifacts": _artifact_truth_from_row(row),
            }
    return answers_by_sample


def _make_task_sample(
    *,
    task_type: str,
    record_id: str,
    media_abs: Path,
    media_rel: Path,
    modality: str,
    prompt: str,
    label: str = "",
) -> TaskSample:
    return TaskSample(
        task=task_type,
        sample_id=record_id,
        modality=modality,
        prompt=prompt,
        fake_path=str(media_abs),
        relative_fake_path=str(media_rel),
        label=label or "",
        file_sha256=None,
        media_meta=None,
    )


def load_oeq_batches(
    *,
    data_root: Path,
    split: str,
    task_type: str,
    allowed_modalities: Sequence[str],
    max_samples: Optional[int],
) -> List[BlindBatch]:
    if task_type not in {"typeb_oeq", "typea_oeq"}:
        raise ValueError(f"Unsupported OEQ task type: {task_type}")

    oeq_root = data_root / "OEQ"
    split_root = oeq_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split not found: {split_root}")

    allowed = set(normalize_modalities(allowed_modalities))
    prompts = OEQ_PROMPTS[task_type]
    answers_by_sample = _load_oeq_answers(split_root)

    batches: List[BlindBatch] = []
    loaded = 0
    for modality in ("audio", "image", "video"):
        if modality not in allowed:
            continue
        manifest_path = split_root / f"manifest_{modality}.csv"
        if not manifest_path.exists():
            continue

        items: List[BlindItem] = []
        for row in _read_csv_rows(manifest_path):
            if max_samples is not None and loaded >= max_samples:
                break
            sample_id = str(row.get("sample_id") or "").strip()
            media_path = str(row.get("media_path") or "").strip()
            track_id = str(row.get("track_id") or "").strip()
            if not sample_id or not media_path:
                continue
            media_rel = Path("OEQ") / split / media_path
            media_abs = data_root / media_rel
            if not media_abs.exists():
                continue
            answer_meta = answers_by_sample.get(sample_id, {})
            label = str(answer_meta.get("label") or "")
            sample = _make_task_sample(
                task_type=task_type,
                record_id=sample_id,
                media_abs=media_abs,
                media_rel=media_rel,
                modality=modality,
                prompt=prompts[modality],
                label=label,
            )
            items.append(
                BlindItem(
                    sample=sample,
                    record_id=sample_id,
                    metadata={
                        "task_type": task_type,
                        "split": split,
                        "sample_id": sample_id,
                        "track_id": track_id,
                        "media_path": str(media_rel),
                        "manifest_file": str(manifest_path),
                        "label": label or None,
                        "ground_truth_artifacts": answer_meta.get("ground_truth_artifacts"),
                    },
                )
            )
            loaded += 1
        if items:
            batches.append(
                BlindBatch(
                    task_type=task_type,
                    split=split,
                    source_path=manifest_path,
                    items=items,
                )
            )
        if max_samples is not None and loaded >= max_samples:
            break
    return batches


def _load_question_answers(path: Path) -> Dict[str, object]:
    answers: Dict[str, object] = {}
    if not path.exists():
        return answers
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or "").strip()
            if question_id:
                answers[question_id] = row.get("ground_truth")
    return answers


def _resolve_question_files(question_dir: Path, question_files: Optional[Sequence[str]]) -> List[Path]:
    if question_files:
        paths: List[Path] = []
        for entry in question_files:
            candidate = Path(entry)
            if not candidate.is_absolute():
                candidate = question_dir / candidate
            if candidate.exists() and candidate.suffix == ".json":
                paths.append(candidate.resolve())
        return sorted(paths)
    return sorted(path for path in question_dir.glob("*.json") if path.is_file())


def _question_modality(raw: object) -> Optional[str]:
    return MODALITY_MAP.get(str(raw or "").strip().lower())


def load_choice_batches(
    *,
    data_root: Path,
    split: str,
    task_type: str,
    allowed_modalities: Sequence[str],
    max_samples: Optional[int],
    question_files: Optional[Sequence[str]] = None,
) -> List[BlindBatch]:
    if task_type not in {"mcq", "tfq"}:
        raise ValueError(f"Unsupported question task type: {task_type}")

    package_root = data_root / ("MCQ" if task_type == "mcq" else "TFQ") / split
    if not package_root.exists():
        raise FileNotFoundError(f"Split not found: {package_root}")

    allowed = set(normalize_modalities(allowed_modalities))
    answers_by_question = _load_question_answers(package_root / "answers.jsonl")

    batches: List[BlindBatch] = []
    loaded = 0
    for json_path in _resolve_question_files(package_root, question_files):
        rows = _read_json(json_path)
        if not isinstance(rows, list):
            continue
        items: List[BlindItem] = []
        for index, row in enumerate(rows):
            if max_samples is not None and loaded >= max_samples:
                break
            if not isinstance(row, dict):
                continue
            modality = _question_modality(row.get("modality"))
            if modality not in allowed:
                continue
            question_id = str(row.get("question_id") or "").strip()
            source_sample_id = str(row.get("sample_id") or "").strip()
            media_path = str(row.get("media_path") or "").strip()
            question = str(row.get("question") or "")
            if not question_id or not media_path or not question:
                continue
            media_rel = Path(media_path)
            media_abs = data_root / media_rel
            if not media_abs.exists():
                continue
            sample = _make_task_sample(
                task_type=task_type,
                record_id=question_id,
                media_abs=media_abs,
                media_rel=media_rel,
                modality=modality,
                prompt=strip_modality_tags(question),
            )
            metadata = {
                "task_type": task_type,
                "split": split,
                "question_id": question_id,
                "sample_id": source_sample_id,
                "media_path": str(media_rel),
                "question_file": str(json_path),
                "question_index": index,
                "raw_question": question,
                "question_type": row.get("question_type"),
                "artifact_type": row.get("artifact_type"),
                "ground_truth": answers_by_question.get(question_id),
            }
            if task_type == "mcq":
                metadata["options"] = row.get("options")
            else:
                metadata["artifact"] = row.get("artifact")
                metadata["location"] = row.get("location")
            items.append(BlindItem(sample=sample, record_id=question_id, metadata=metadata))
            loaded += 1
        if items:
            batches.append(
                BlindBatch(
                    task_type=task_type,
                    split=split,
                    source_path=json_path,
                    items=items,
                )
            )
        if max_samples is not None and loaded >= max_samples:
            break
    return batches


def load_batches(
    *,
    data_root: Path,
    task_type: str,
    split: str,
    allowed_modalities: Sequence[str],
    max_samples: Optional[int],
    question_files: Optional[Sequence[str]] = None,
) -> List[BlindBatch]:
    normalized_task = normalize_task_name(task_type)
    if normalized_task in {"typeb_oeq", "typea_oeq"}:
        return load_oeq_batches(
            data_root=data_root,
            split=split,
            task_type=normalized_task,
            allowed_modalities=allowed_modalities,
            max_samples=max_samples,
        )
    return load_choice_batches(
        data_root=data_root,
        split=split,
        task_type=normalized_task,
        allowed_modalities=allowed_modalities,
        max_samples=max_samples,
        question_files=question_files,
    )
