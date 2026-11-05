"""Load Ardook's facts from the machine-readable block in docs/ardook-profile.md.

The profile is the single source of truth, so no Ardook facts are written in code.
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "docs" / "ardook-profile.md"


def load_profile(path: Path = PROFILE_PATH) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"```yaml\n(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError(f"No ```yaml facts block found in {path}")
    return yaml.safe_load(match.group(1))
