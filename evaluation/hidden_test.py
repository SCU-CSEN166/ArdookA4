"""Turn pipeline runs on the TA's hidden test into hidden_test_outputs.csv (grant_id,label,reason).

The TA's question: is the grant relevant for Ardook, meaning Ardook can apply AND it fits Ardook's work?
Threshold used here (stated in the README too):
    Relevant      = eligibility is Lead or Conditional lead (not ruled out)  AND  Fit is High or Moderate
    Not relevant  = ruled out by a confirmed requirement (location, applicant type, AI focus, ...)  OR  Fit is Low
Deadlines and closed status are ignored, as the TA asked.

Usage (from the repo root, after running src/run_pipeline.py on hiddenTest/Ardook_hidden_test_grants.xlsx):
  python evaluation/hidden_test.py --eligibility-run outputs/Ardook_hidden_test_grants_nuextract
  python evaluation/hidden_test.py --eligibility-run outputs/Ardook_hidden_test_grants_api_qwen3-32b \
      --fit-run outputs/Ardook_hidden_test_grants_api_qwen3-32b_direct
  python evaluation/hidden_test.py ... --labels hiddenTest/expected_labels.csv   # once the TA shares them

--fit-run lets a "rate it directly" run supply Fit while the checklist run supplies eligibility.
Without --out, it prints a comparison of every hidden-test run in outputs/ as well.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs"
STEM = "Ardook_hidden_test_grants"


def read(run: Path) -> dict[str, dict]:
    return {r["id"]: r for r in csv.DictReader((run / "results.csv").open(encoding="utf-8"))}


def decide(elig: dict, fit: dict) -> tuple[str, str]:
    if elig["eligibility"] == "Not a lead":
        return "Not relevant", f"Can't apply: {elig['eligibility_why']}"
    if fit["fit"] == "Low":
        return "Not relevant", f"Poor fit: {fit['why'].removeprefix('Low: ')}"
    reason = f"{fit['fit']} fit: {fit['why'].split(': ', 1)[-1].rstrip('. ')}"
    if elig["eligibility"] == "Conditional lead":
        reason += f". Still to confirm: {elig['check'].replace('if ', '', 1)}"
    return "Relevant", reason


GOLD_MAP = {"good": "Relevant", "relevant": "Relevant", "bad": "Not relevant", "not relevant": "Not relevant"}


def read_labels(path: Path) -> dict[str, str]:
    """The TA's expected labels, CSV or XLSX, with grant_id and label (Good/Bad or Relevant/Not relevant)."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd

        rows = pd.read_excel(path, dtype=str).to_dict("records")
    else:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {str(r["grant_id"]).strip(): GOLD_MAP[str(r["label"]).strip().lower()] for r in rows}


def score(predicted: dict[str, str], labels_path: Path) -> None:
    gold = read_labels(labels_path)
    ids = [i for i in predicted if i in gold]
    tp = sum(predicted[i] == gold[i] == "Relevant" for i in ids)
    fp = sum(predicted[i] == "Relevant" != gold[i] for i in ids)
    fn = sum(gold[i] == "Relevant" != predicted[i] for i in ids)
    acc = sum(predicted[i] == gold[i] for i in ids) / len(ids)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"\nvs. expected labels (n={len(ids)}, Relevant = positive): "
          f"accuracy={acc:.2f} precision={p:.2f} recall={r:.2f} F1={f1:.2f}")
    for i in ids:
        if predicted[i] != gold[i]:
            print(f"  grant {i}: predicted {predicted[i]}, expected {gold[i]}")


def compare_all() -> None:
    runs = sorted(p.parent for p in OUT.glob(f"{STEM}_*/results.csv"))
    checklist = [r for r in runs if "_direct" not in r.name]
    if not runs:
        return
    print("\nAll hidden-test runs (R = Relevant, - = Not relevant; direct runs use the first checklist run's eligibility):")
    base = read(checklist[0]) if checklist else None
    table = {}
    for run in runs:
        rows = read(run)
        elig = rows if "_direct" not in run.name else base
        if elig is None:
            continue
        table[run.name.removeprefix(STEM + "_")] = {i: decide(elig[i], rows[i])[0] for i in rows}
    ids = sorted(next(iter(table.values())), key=int)
    print(f"{'system':32}" + "".join(f"{i:>4}" for i in ids))
    for name, labels in table.items():
        print(f"{name:32}" + "".join(f"{'R' if labels[i] == 'Relevant' else '-':>4}" for i in ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eligibility-run", type=Path, required=True, help="checklist run (has eligibility)")
    parser.add_argument("--fit-run", type=Path, help="run that supplies Fit (default: the eligibility run)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "hidden_test_outputs.csv")
    parser.add_argument("--labels", type=Path, help="CSV with grant_id,label from the TA, to score the output")
    args = parser.parse_args()

    elig, fit = read(args.eligibility_run), read(args.fit_run or args.eligibility_run)
    if "eligibility" not in next(iter(elig.values())) or not next(iter(elig.values()))["eligibility"]:
        sys.exit("--eligibility-run must be a checklist run (not --direct)")
    rows = []
    for gid in sorted(elig, key=int):
        label, reason = decide(elig[gid], fit[gid])
        rows.append({"grant_id": gid, "label": label, "reason": reason})
        print(f"{gid:>3} {label:<13} {elig[gid]['program'][:45]:<45} {reason[:110]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["grant_id", "label", "reason"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out}")
    if args.labels:
        score({r["grant_id"]: r["label"] for r in rows}, args.labels)
    compare_all()


if __name__ == "__main__":
    main()
