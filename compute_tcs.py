from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
from pathlib import Path
from typing import List

from evaluate_predictions import (
    SUMMARY_RESULTS_DIRNAME,
    TCS_WEIGHTS,
    compute_tcs_from_tasks,
)
from starter_kit._bootstrap import DEFAULT_OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TCS from per-model evaluation summaries.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / SUMMARY_RESULTS_DIRNAME,
        help="Directory containing per-model evaluation summary JSON files.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional single summary JSON file to update.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory. When omitted, summary files are updated in place.",
    )
    return parser.parse_args()


def _iter_summary_paths(args: argparse.Namespace) -> List[Path]:
    if args.summary is not None:
        return [args.summary.resolve()]
    root = args.results_root.resolve()
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.json") if path.is_file())


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _destination_path(path: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return path
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root / path.name


def main() -> None:
    args = parse_args()
    summary_paths = _iter_summary_paths(args)
    if not summary_paths:
        raise SystemExit("No summary JSON files found.")

    written = []
    output_root = args.output_root.resolve() if args.output_root else None
    for path in summary_paths:
        payload = _load_summary(path)
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            continue
        payload["tcs_weights"] = dict(TCS_WEIGHTS)
        payload["tcs_formula"] = (
            "tcs = 0.4 * (100 * acc_det) + 0.3 * (100 * ((typea_f_0_5 + typeb_f_0_5) / 2)) "
            "+ 0.3 * (100 * (0.5 * acc_tfq + 0.5 * score_mcq))"
        )
        payload["tcs_by_modality"] = compute_tcs_from_tasks(tasks)
        destination = _destination_path(path, output_root)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append({"source": str(path), "output": str(destination)})

    print(json.dumps({"written_summaries": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
