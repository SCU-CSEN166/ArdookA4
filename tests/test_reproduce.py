"""The reproduction kit (reproduce/): GPU -> model choice, and the bundled answers still give the bundled numbers.

No model or server: the checklist replay reads the bundled Qwen3-32B answers (the judge step needs MiniLM,
so it's checked by running the notebook, not here).
"""

import csv
import sys
from pathlib import Path

import pytest

from ardook_grants.backends import AnswerCache
from ardook_grants.checklist import build_groups
from ardook_grants.data import load_any
from ardook_grants.metrics import fit_scores
from ardook_grants.pipeline import run
from ardook_grants.profile import ARDOOK_DATA_PROFILE, load_profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduce"))
import repro  # noqa: E402


@pytest.mark.parametrize("memory_gb, capability, model", [
    (80, 8.0, "Qwen/Qwen3-32B"), (94, 9.0, "Qwen/Qwen3-32B"),
    (40, 8.0, "Qwen/Qwen3-32B-AWQ"), (48, 8.6, "Qwen/Qwen3-32B-AWQ"),
    (24, 8.9, "Qwen/Qwen3-4B"), (15, 7.5, "Qwen/Qwen3-4B"), (32, 7.0, "Qwen/Qwen3-4B"),
    (8, 8.6, None), (None, 0, None),
])
def test_pick_model_by_gpu_memory(memory_gb, capability, model):
    assert repro.pick_model(memory_gb, capability)["model"] == model


def test_bundle_is_complete_and_small():
    for name in repro.DATA_FILES:
        assert (repro.BUNDLE / "data" / name).exists()
    for step in ("checklist", "judge"):
        assert (repro.BUNDLE / "cache" / f"{step}__qwen-qwen3-32b.json").exists()
    assert sum(p.stat().st_size for p in repro.BUNDLE.rglob("*") if p.is_file()) < 5_000_000
    with (repro.BUNDLE / "data" / "real_grants.csv").open(encoding="utf-8") as f:
        assert "Applied?" not in next(csv.reader(f))


def test_checklist_replay_matches_expected_results():
    profile = load_profile(ARDOOK_DATA_PROFILE)
    groups = build_groups(profile)
    rows = profile["labeled_sheet_rows"]
    cache = AnswerCache(repro.BUNDLE / "cache" / "checklist__qwen-qwen3-32b.json")
    expected = {(r["dataset"], r["system"]): r for r in repro.expected()}
    for ds in repro.DATASETS:
        path = repro.BUNDLE / "data" / Path(ds["input"]).name
        grants = load_any(path, ds["split"], label_rows=(rows[0], rows[1]))
        grants = [g for g in grants if g.get("gold_fit")] if ds["labeled_only"] else grants
        results = run(grants, groups, profile, None, cache, rules_only=True, verbose=False)
        assert len(results) == len(grants), f"{ds['key']}: some answers are missing from the bundle"
        scores = fit_scores([r.gold_fit for r in results], [r.fit for r in results])
        want = expected[(ds["key"], "checklist alone")]
        assert scores["accuracy"] == pytest.approx(float(want["3-tier accuracy"]), abs=5e-4), ds["key"]
        assert scores["f1"] == pytest.approx(float(want["3-tier macro F1"]), abs=5e-4), ds["key"]
