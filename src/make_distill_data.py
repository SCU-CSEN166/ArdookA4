"""Export a fine-tuning set that teaches a small model (e.g. Qwen3-4B) to rate grants like the Qwen3-32B judge.

Input: a run_judge.py run on data/synthetic_ardook_data_train.csv (the teacher), plus that file's known labels.
Only answers where the teacher's rating equals the known label are kept, so the student learns from correct
ratings with the teacher's reasons. The student sees no retrieved examples (it has to learn the rating policy
itself, and a short prompt keeps it fast on a laptop), only the judge instructions, the profile and the listing.

Never include data/synthetic_ardook_data.csv's test split or data/synthetic_ardook_data_alt.csv: they are the
in-distribution and held-out test sets.

Usage (from the repo root):
  python src/make_distill_data.py --teacher-run outputs/synthetic_ardook_data_train_judge_k16_v1_ardook-data
Writes data/distill/train.jsonl and valid.jsonl ({"messages": [system, user, assistant]}: the chat format
mlx-lm's LoRA trainer and TRL's SFTTrainer read).
"""

import argparse
import csv
import json
import random
from pathlib import Path

from ardook_grants.data import load_any
from ardook_grants.judge import JUDGE_INSTRUCTIONS, judge_profile_text, judge_prompt
from ardook_grants.profile import ARDOOK_DATA_PROFILE, REPO_ROOT

# The student isn't shown examples, so drop the instruction sentence that refers to them.
STUDENT_INSTRUCTIONS = JUDGE_INSTRUCTIONS.replace(
    "The examples are similar grants the organization's staff already rated; follow how they rate, but judge this grant on its own text.\n", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "synthetic_ardook_data_train.csv")
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "distill")
    parser.add_argument("--valid-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=166)
    args = parser.parse_args()

    grants = {g["id"]: g for g in load_any(args.input)}
    teacher = list(csv.DictReader((args.teacher_run / "results.csv").open(encoding="utf-8")))
    profile_text = judge_profile_text(ARDOOK_DATA_PROFILE)

    rows, disagreed = [], 0
    for t in teacher:
        g = grants[int(t["id"])]
        if t["fit"] != g["gold_fit"] or not t["why"].strip():
            disagreed += 1
            continue
        rows.append({"messages": [
            {"role": "system", "content": STUDENT_INSTRUCTIONS},
            {"role": "user", "content": judge_prompt(g["text"], profile_text, [])},
            {"role": "assistant", "content": json.dumps({"fit": t["fit"], "reason": t["why"]}, ensure_ascii=False)},
        ]})
    random.Random(args.seed).shuffle(rows)
    n_valid = max(1, int(len(rows) * args.valid_share))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("valid.jsonl", rows[:n_valid]), ("train.jsonl", rows[n_valid:])):
        with (args.out / name).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    labels = {lab: sum(json.loads(r["messages"][2]["content"])["fit"] == lab for r in rows) for lab in ("High", "Moderate", "Low")}
    print(f"Kept {len(rows)} of {len(teacher)} teacher answers ({disagreed} disagreed with the known label); "
          f"labels {labels}; wrote {len(rows) - n_valid} train / {n_valid} valid to {args.out}")


if __name__ == "__main__":
    main()
