"""Build reproduce/bundle/ from a machine that has the team's data/ and outputs/cache/ (maintainers only).

  python reproduce/build_bundle.py

Writes:
  bundle/data/          the three labeled sets (real_grants.csv without the "Applied?" column, which the
                        pipeline never reads; the synthetic sets use fictional funders)
  bundle/cache/         only the Qwen3-32B answers those sets need (no hidden-test answers)
  bundle/expected_results.csv   scores from replaying those answers through the current code

Judge answers missing from outputs/cache/ (lost when two runs saved the same cache file at once) are restored
from that run's own trace.jsonl, which records the model's raw answer per grant.
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro  # noqa: E402

from ardook_grants.backends import AnswerCache  # noqa: E402
from ardook_grants.checklist import build_groups  # noqa: E402
from ardook_grants.data import load_any  # noqa: E402
from ardook_grants.judge import (SPONSOR_CSV, SYNTHETIC_ARDOOK_DATA, Judge, Retriever, default_pool,  # noqa: E402
                                 judge_profile_text)
from ardook_grants.pipeline import GROUP_ORDER  # noqa: E402
from ardook_grants.profile import ARDOOK_DATA_PROFILE, load_profile  # noqa: E402

ROOT, BUNDLE = repro.ROOT, repro.BUNDLE
FULL_CACHE = {"checklist": ROOT / "outputs" / "cache" / "api__qwen-qwen3-32b.json",
              "judge": ROOT / "outputs" / "cache" / "judge__qwen-qwen3-32b.json"}
# The reported k=16 judge run per dataset, whose trace can fill answers missing from the cache.
JUDGE_RUNS = {"real": "real_grants_judge_k16_v1_ardook-data",
              "synthetic_test": "synthetic_ardook_data_test_judge_k16_v1_ardook-data",
              "synthetic_alt_test": "synthetic_ardook_data_alt_test_judge_k16_v1_ardook-data"}


def copy_data() -> None:
    (BUNDLE / "data").mkdir(parents=True, exist_ok=True)
    with (ROOT / "data" / "real_grants.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    drop = rows[0].index("Applied?")
    with (BUNDLE / "data" / "real_grants.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([c for i, c in enumerate(r) if i != drop] for r in rows)
    for name in repro.DATA_FILES[1:]:
        shutil.copy(ROOT / "data" / name, BUNDLE / "data" / name)


def main() -> None:
    copy_data()
    full = {step: AnswerCache(path) for step, path in FULL_CACHE.items()}
    keep = {"checklist": {}, "judge": {}}

    profile = load_profile(ARDOOK_DATA_PROFILE)
    groups = build_groups(profile)
    rows = profile["labeled_sheet_rows"]
    pool = default_pool(repro.PROFILE, SPONSOR_CSV, (rows[0], rows[1]), SYNTHETIC_ARDOOK_DATA)
    judge = Judge(None, full["judge"], judge_profile_text(ARDOOK_DATA_PROFILE), Retriever(pool), k=repro.JUDGE_K)

    for ds in repro.DATASETS:
        grants = load_any(ROOT / ds["input"], ds["split"], label_rows=(rows[0], rows[1]))
        grants = [g for g in grants if g.get("gold_fit")] if ds["labeled_only"] else grants
        trace_path = ROOT / "outputs" / JUDGE_RUNS[ds["key"]] / "trace.jsonl"
        traced = {}
        if trace_path.exists():
            traced = {str(t["id"]): t["raw"] for t in map(json.loads, trace_path.read_text(encoding="utf-8").splitlines())}
        restored = missing = 0
        for g in grants:
            for name in GROUP_ORDER:
                key = AnswerCache.key(name, groups[name].version, g["text"])
                if key in full["checklist"].data:
                    keep["checklist"][key] = full["checklist"].data[key]
            for version, user, _ in judge.requests(g["text"]):
                key = AnswerCache.key("judge", version, user)
                if key in full["judge"].data:
                    keep["judge"][key] = full["judge"].data[key]
                elif str(g["id"]) in traced:
                    keep["judge"][key] = {"answer": traced[str(g["id"])], "seconds": 0.0}
                    restored += 1
                else:
                    missing += 1
        print(f"{ds['key']}: {len(grants)} grants, {restored} judge answers restored from trace, {missing} missing")

    (BUNDLE / "cache").mkdir(parents=True, exist_ok=True)
    for step, data in keep.items():
        path = BUNDLE / "cache" / f"{step}__qwen-qwen3-32b.json"
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {len(data)} answers")

    # Score by replaying the bundle exactly as the notebook does.
    out = ROOT / "outputs" / "reproduce" / "bundle_check"
    caches = repro.replay_caches(out / "cache")
    table = []
    for ds in repro.DATASETS:
        dirs = repro.run_system(ds, out, caches, replay=True)
        for row in repro.score(dirs):
            table.append({"dataset": ds["key"], **row})
    with (BUNDLE / "expected_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    subprocess.run([sys.executable, "-c", "import csv,sys; [print(r) for r in csv.reader(open(sys.argv[1]))]",
                    BUNDLE / "expected_results.csv"])


if __name__ == "__main__":
    main()
