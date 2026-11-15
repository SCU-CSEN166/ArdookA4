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


SUMMARY_SECTIONS = ("At a glance", "Mission and the problem it solves", "How it works", "What it builds",
                    "Theme ratings (used for Fit)")
ARDOOK_DATA_PROFILE = REPO_ROOT / "docs" / "ardook-data-profile.md"


def profile_summary(path: Path = PROFILE_PATH) -> str:
    """The profile's readable sections that describe Ardook, without source numbers, for prompting a model."""
    text = Path(path).read_text(encoding="utf-8")
    parts = []
    for title in SUMMARY_SECTIONS:
        match = re.search(rf"^#+ {re.escape(title)}\n(.*?)(?=^#+ |\Z)", text, re.DOTALL | re.MULTILINE)
        if match:
            body = re.sub(r"\s*\[\d+\]", "", match.group(1)).strip()
            parts.append(f"## {title}\n{body}")
    return "\n\n".join(parts)
