from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from starter_kit._bootstrap import DEFAULT_DATA_ROOT


TASKS = ("typeb_oeq", "typea_oeq", "mcq", "tfq")
TASK_FILE_NAMES = {task: f"{task}.jsonl" for task in TASKS}
TASK_ID_FIELDS = {
    "typeb_oeq": "sample_id",
    "typea_oeq": "sample_id",
    "mcq": "question_id",
    "tfq": "question_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a participant submission directory or zip archive.")
    parser.add_argument("--submission", type=Path, required=True, help="Submission directory or .zip archive.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="private_test", choices=("private_test", "public_val", "train"))
    parser.add_argument("--task", type=str, default="all", choices=(*TASKS, "all"))
    parser.add_argument(
        "--allow-extra-keys",
        action="store_true",
        help="Do not warn when records contain keys beyond the required id field and response.",
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def _selected_tasks(task: str) -> List[str]:
    return list(TASKS) if task == "all" else [task]


def _load_expected_oeq_ids(data_root: Path, split: str) -> List[str]:
    expected: List[str] = []
    seen: set[str] = set()
    for modality in ("audio", "image", "video"):
        path = data_root / "OEQ" / split / f"manifest_{modality}.csv"
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sample_id = str(row.get("sample_id") or "").strip()
                if sample_id and sample_id not in seen:
                    expected.append(sample_id)
                    seen.add(sample_id)
    return expected


def _load_expected_question_ids(data_root: Path, split: str, package_name: str) -> List[str]:
    expected: List[str] = []
    seen: set[str] = set()
    for json_path in sorted((data_root / package_name / split).glob("*.json")):
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            question_id = str(row.get("question_id") or "").strip()
            if question_id and question_id not in seen:
                expected.append(question_id)
                seen.add(question_id)
    return expected


def load_expected_ids(data_root: Path, split: str, task: str) -> List[str]:
    if task in {"typeb_oeq", "typea_oeq"}:
        return _load_expected_oeq_ids(data_root, split)
    if task == "mcq":
        return _load_expected_question_ids(data_root, split, "MCQ")
    if task == "tfq":
        return _load_expected_question_ids(data_root, split, "TFQ")
    raise ValueError(f"Unsupported task: {task}")


def _resolve_zip_member(archive: zipfile.ZipFile, filename: str) -> Optional[str]:
    candidates = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and Path(info.filename).name == filename
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise ValueError(f"Archive contains multiple candidates for {filename}: {candidates}")


def _open_submission_text(submission: Path, filename: str):
    if submission.is_dir():
        path = submission / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing submission file: {path}")
        return path.open("r", encoding="utf-8"), str(path)
    if submission.is_file() and submission.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(submission)
        member_name = _resolve_zip_member(archive, filename)
        if member_name is None:
            archive.close()
            raise FileNotFoundError(f"Missing {filename} in archive {submission}")
        raw = archive.open(member_name, "r")
        text = io.TextIOWrapper(raw, encoding="utf-8")
        return _ZipTextHandle(archive=archive, wrapper=text), f"{submission}:{member_name}"
    raise ValueError(f"Unsupported submission path: {submission}")


class _ZipTextHandle:
    def __init__(self, *, archive: zipfile.ZipFile, wrapper: io.TextIOWrapper):
        self._archive = archive
        self._wrapper = wrapper

    def __enter__(self) -> io.TextIOWrapper:
        return self._wrapper

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._wrapper.close()
        finally:
            self._archive.close()


def _iter_jsonl_records(handle: Iterable[str]) -> Iterator[Tuple[int, dict]]:
    for line_no, line in enumerate(handle, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise TypeError(f"Line {line_no} is not a JSON object.")
        yield line_no, payload


def validate_task_file(
    *,
    submission: Path,
    task: str,
    expected_ids: Sequence[str],
    allow_extra_keys: bool,
) -> dict:
    filename = TASK_FILE_NAMES[task]
    id_field = TASK_ID_FIELDS[task]
    expected_set = set(expected_ids)
    seen_ids: set[str] = set()
    submitted_ids: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []
    extra_key_count = 0
    empty_response_count = 0
    record_count = 0

    try:
        context, resolved_path = _open_submission_text(submission, filename)
    except Exception as exc:  # noqa: BLE001
        return {
            "task": task,
            "file": filename,
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
        }

    with context as handle:
        try:
            for line_no, record in _iter_jsonl_records(handle):
                record_count += 1
                raw_id = record.get(id_field)
                record_id = str(raw_id or "").strip()
                if not record_id:
                    errors.append(f"{filename}:{line_no}: missing or empty `{id_field}`.")
                    continue
                response = record.get("response")
                if not isinstance(response, str):
                    errors.append(f"{filename}:{line_no}: `response` must be a string.")
                    continue
                if not response.strip():
                    empty_response_count += 1
                extra_keys = sorted(set(record.keys()) - {id_field, "response"})
                if extra_keys and not allow_extra_keys:
                    extra_key_count += 1
                if record_id in seen_ids:
                    errors.append(f"{filename}:{line_no}: duplicate id `{record_id}`.")
                    continue
                seen_ids.add(record_id)
                submitted_ids.append(record_id)
        except json.JSONDecodeError as exc:
            errors.append(f"{filename}: invalid JSONL content: {exc}")
        except TypeError as exc:
            errors.append(f"{filename}: {exc}")

    submitted_set = set(submitted_ids)
    missing_ids = sorted(expected_set - submitted_set)
    extra_ids = sorted(submitted_set - expected_set)

    if missing_ids:
        errors.append(f"{filename}: missing {len(missing_ids)} required ids.")
    if extra_ids:
        errors.append(f"{filename}: found {len(extra_ids)} unexpected ids.")
    if extra_key_count:
        warnings.append(f"{filename}: {extra_key_count} record(s) contain extra keys beyond `{id_field}` and `response`.")
    if empty_response_count:
        warnings.append(f"{filename}: {empty_response_count} record(s) contain empty responses.")

    return {
        "task": task,
        "file": filename,
        "resolved_path": resolved_path,
        "valid": not errors,
        "expected_records": len(expected_ids),
        "submitted_records": record_count,
        "unique_ids": len(submitted_set),
        "missing_count": len(missing_ids),
        "extra_count": len(extra_ids),
        "empty_response_count": empty_response_count,
        "errors": errors,
        "warnings": warnings,
        "missing_examples": missing_ids[:5],
        "extra_examples": extra_ids[:5],
    }


def main() -> None:
    args = parse_args()
    submission = args.submission.resolve()
    data_root = args.data_root.resolve()

    results = []
    for task in _selected_tasks(args.task):
        expected_ids = load_expected_ids(data_root, args.split, task)
        results.append(
            validate_task_file(
                submission=submission,
                task=task,
                expected_ids=expected_ids,
                allow_extra_keys=bool(args.allow_extra_keys),
            )
        )

    summary = {
        "submission": str(submission),
        "data_root": str(data_root),
        "split": args.split,
        "tasks": results,
        "valid": all(result["valid"] for result in results),
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n", encoding="utf-8")
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
