from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
STARTER_KIT_ROOT = PACKAGE_ROOT.parent
IMPORT_ROOT = STARTER_KIT_ROOT

DATA_ROOT_CANDIDATE_NAMES = ("tridf-blind", "trident", "data_blind")


def _resolve_repo_root() -> Path:
    for candidate_root in (STARTER_KIT_ROOT, STARTER_KIT_ROOT.parent):
        for data_dir_name in DATA_ROOT_CANDIDATE_NAMES:
            if (candidate_root / data_dir_name).exists():
                return candidate_root
    return STARTER_KIT_ROOT


def _resolve_default_data_root(repo_root: Path) -> Path:
    for data_dir_name in DATA_ROOT_CANDIDATE_NAMES:
        candidate = repo_root / data_dir_name
        if candidate.exists():
            return candidate
    return repo_root / DATA_ROOT_CANDIDATE_NAMES[0]


REPO_ROOT = _resolve_repo_root()

DEFAULT_DATA_ROOT = _resolve_default_data_root(REPO_ROOT)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "starter_kit_outputs"
DEFAULT_EVAL_CACHE_ROOT = REPO_ROOT / "starter_kit_eval_cache"
DEFAULT_MODEL_CACHE_DIR = REPO_ROOT / "models"

text = str(IMPORT_ROOT)
if text not in sys.path:
    sys.path.insert(0, text)
