"""Score "checklist eligibility + another run's Fit" against gold labels, for any labeled input.

The combination used for the final system:
    Fit = Low                 if the checklist rules the grant out (a confirmed hard requirement fails)
    Fit = the Fit run's tier  otherwise
    Relevant = Fit is High or Moderate   (the TA's hidden-test question)

Usage (from the repo root):
  python evaluation/combined_eval.py --eligibility-run outputs/synthetic_ardook_data_test_api_qwen3-32b_ardook-data \
      --fit-run outputs/synthetic_ardook_data_test_judge_k4_v5_ardook-data outputs/synthetic_ardook_data_test_judge_k0_v1_ardook-data
Prints 3-tier scores and relevance (binary) accuracy / precision / recall / F1 for each Fit run, alone and combined.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ardook_grants.metrics import fit_scores  # noqa: E402


def read(run: Path) -> dict[str, dict]:
    return {r["id"]: r for r in csv.DictReader((Path(run) / "results.csv").open(encoding="utf-8"))}


def binary(gold: list[str], pred: list[str]) -> dict:
    g = [x != "Low" for x in gold]
    p = [x != "Low" for x in pred]
    tp = sum(a and b for a, b in zip(g, p))
    fp = sum(b and not a for a, b in zip(g, p))
    fn = sum(a and not b for a, b in zip(g, p))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"acc": sum(a == b for a, b in zip(g, p)) / len(g), "p": prec, "r": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def line(name: str, gold: list[str], pred: list[str]) -> str:
    s, b = fit_scores(gold, pred), binary(gold, pred)
    return (f"{name:58s} n={s['n']:<4} 3-tier acc={s['accuracy']:.3f} F1={s['f1']:.3f} [{s['f1_95ci']}]   "
            f"relevant acc={b['acc']:.3f} P={b['p']:.3f} R={b['r']:.3f} F1={b['f1']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eligibility-run", type=Path, required=True)
    parser.add_argument("--fit-run", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    elig = read(args.eligibility_run)
    ids = [i for i, r in elig.items() if r.get("gold_fit")]
    gold = [elig[i]["gold_fit"] for i in ids]
    print(line(f"checklist alone ({args.eligibility_run.name})", gold, [elig[i]["fit"] for i in ids]))
    for run in args.fit_run:
        fit = read(run)
        common = [i for i in ids if i in fit]
        g = [elig[i]["gold_fit"] for i in common]
        alone = [fit[i]["fit"] for i in common]
        combined = ["Low" if elig[i]["eligibility"] == "Not a lead" else fit[i]["fit"] for i in common]
        print(line(f"  {run.name} alone", g, alone))
        print(line(f"  checklist eligibility + {run.name}", g, combined))


if __name__ == "__main__":
    main()
