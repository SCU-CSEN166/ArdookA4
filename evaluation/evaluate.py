"""Compare pipeline runs on the labeled grants and write the files the Week 8 write-up needs.

Usage (from the repo root, after src/run_pipeline.py has produced some runs):
  python evaluation/evaluate.py                               # every outputs/real_grants_* run
  python evaluation/evaluate.py --stem synthetic_grants_test  # every synthetic test-split run
  python evaluation/evaluate.py --runs outputs/real_grants_nuextract outputs/real_grants_llm_qwen3-4b

For each run it scores Fit against the labels (all labeled grants, and the fully real ones when the
sheet marks some rows as partly made up), next to an always-Low floor. For checklist runs it also
trains a logistic regression on the checklist answers (5-fold cross-validation, 5 shuffles): the
learned alternative to the fixed Fit rule.

Writes:
  results/pipeline_comparison.csv          metrics only, safe to commit
  outputs/eval/errors_<run>.csv            each wrong grant with its trace, most harmful first
  outputs/eval/eligibility_to_check.csv    the pipeline's eligibility calls, for two people to check by hand
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ardook_grants.metrics import LABELS, fit_scores, format_scores  # noqa: E402
from ardook_grants.profile import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs"
SEED = 0
HARM = {("High", "Low"): "1 missed a High (rated Low)", ("High", "Moderate"): "2 High rated Moderate",
        ("Low", "High"): "3 Low rated High"}


def load_run(run_dir: Path) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader((run_dir / "results.csv").open(encoding="utf-8")))
    traces = {}
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            traces[str(t["id"])] = t
    return rows, traces


def subsets(rows: list[dict]) -> dict[str, list[dict]]:
    labeled = [r for r in rows if r["gold_fit"]]
    out = {"all labeled": labeled}
    real = [r for r in labeled if r.get("synthetic", "").strip() == "None"]
    if real and len(real) < len(labeled):
        out["fully real"] = real
    return out


def features(trace: dict, row: dict) -> dict:
    """Checklist answers as yes/no features for the learned Fit model."""
    f = {f"theme={row['theme']}": 1, f"entity={row['entity']}": 1}
    facts = trace.get("facts", {})
    for area in facts.get("focus_areas") or []:
        f[f"area={area}"] = 1
    for t in facts.get("applicant_types") or []:
        f[f"applicant={t}"] = 1
    for t in facts.get("funding_types") or []:
        f[f"funding={t}"] = 1
    if facts.get("women_led_required") == "yes":
        f["women_led_required"] = 1
    if facts.get("program_focus_is_ai_or_deep_tech") == "yes":
        f["ai_program"] = 1
    verdict = trace.get("verdict", {})
    geo = set(verdict.get("geo_reasons", [])) | set(verdict.get("geo_conditions", []))
    f["org_conditions"] = sum(c not in geo for c in verdict.get("conditions", []))
    f["org_reasons"] = sum(r not in geo for r in verdict.get("reasons", []))
    return f


def trained_fit(rows: list[dict], traces: dict, repeats: int = 5) -> tuple[list[str], list[float]]:
    """Predictions from the first shuffle, and macro F1 for every shuffle."""
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    X = DictVectorizer(sparse=False).fit_transform([features(traces[r["id"]], r) for r in rows])
    y = np.array([r["gold_fit"] for r in rows])
    runs = []
    for rep in range(repeats):
        pred = np.empty(len(y), dtype=object)
        for train, test in StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(X, y):
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X[train], y[train])
            pred[test] = model.predict(X[test])
        runs.append(list(pred))
    f1s = [f1_score(y, p, labels=LABELS, average="macro", zero_division=0) for p in runs]
    return runs[0], f1s


def table_row(system: str, subset: str, s: dict, extra: dict | None = None) -> dict:
    row = {"system": system, "subset": subset, **{k: v for k, v in s.items() if k != "confusion"}}
    row["confusion"] = json.dumps(s.get("confusion"))
    return {**row, **(extra or {})}


def write_errors(name: str, rows: list[dict], data: dict) -> Path:
    wrong = [r for r in rows if r["gold_fit"] and r["gold_fit"] != r["fit"]]
    for r in wrong:
        r["harm"] = HARM.get((r["gold_fit"], r["fit"]), "4 off by one tier")
    wrong.sort(key=lambda r: (r["harm"], r["program"]))
    path = OUT / "eval" / f"errors_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["harm", "id", "program", "gold_fit", "sheet_fit", "fit", "why", "theme_areas", "entity",
            "eligibility", "eligibility_why", "evidence", "notes", "synthetic", "staff_why", "staff_concern"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in wrong:
            w.writerow({**r, **data.get(r["id"], {})})
    return path


def write_eligibility_checklist(rows: list[dict], data: dict) -> Path:
    path = OUT / "eval" / "eligibility_to_check.csv"
    cols = ["id", "program", "eligibility", "eligibility_why", "check", "staff_concern", "staff_action",
            "checker_1 (Lead / Conditional lead / Not a lead)", "checker_2", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if r["gold_fit"]:
                w.writerow({**r, **data.get(r["id"], {})})
    return path


def load_staff_columns(path: Path) -> dict:
    """The sheet's human reasoning, shown next to errors to help explain them (never fed to the model)."""
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        key = (r.get("Priority / Deadline Order") or "").strip()
        if key.isdigit():
            out[key] = {"staff_why": r.get("Why It Fits Ardook Data", ""), "staff_concern": r.get("Key Concern / Disqualifier", ""),
                        "staff_action": r.get("Recommended Action", "")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, help="output folders from run_pipeline.py")
    parser.add_argument("--stem", default="real_grants", help="evaluate every outputs/<stem>_* folder")
    parser.add_argument("--errors-for", help="which run gets the error report (default: the first checklist run)")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "real_grants.csv",
                        help="Grant Tracker CSV, for the staff's reasoning next to each error")
    args = parser.parse_args()

    runs = args.runs or sorted(p.parent for p in OUT.glob(f"{args.stem}_*/results.csv"))
    if not runs:
        sys.exit(f"No runs found for outputs/{args.stem}_*. Run src/run_pipeline.py first.")
    data = load_staff_columns(args.data)
    table, checklist_runs = [], []

    first_rows, _ = load_run(runs[0])
    for subset, rows in subsets(first_rows).items():
        s = fit_scores([r["gold_fit"] for r in rows], ["Low"] * len(rows))
        print(format_scores(f"always Low [{subset}]", s))
        table.append(table_row("always Low", subset, s))

    for run_dir in runs:
        name = run_dir.name.removeprefix(f"{args.stem}_")
        rows, traces = load_run(run_dir)
        for subset, sub in subsets(rows).items():
            s = fit_scores([r["gold_fit"] for r in sub], [r["fit"] for r in sub])
            print(format_scores(f"{name} [{subset}]", s))
            table.append(table_row(name, subset, s))
        labeled = [r for r in rows if r["gold_fit"]]
        if traces and "_direct" not in name and labeled:
            checklist_runs.append((name, rows))
            pred, f1s = trained_fit(labeled, traces)
            s = fit_scores([r["gold_fit"] for r in labeled], pred)
            print(format_scores(f"{name} + logistic regression (5-fold CV)", s))
            print(f"{'':44s} F1 over 5 shuffles: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
            table.append(table_row(f"{name} + logistic regression (CV)", "all labeled", s,
                                   {"f1_mean_5_shuffles": round(float(np.mean(f1s)), 3),
                                    "f1_sd_5_shuffles": round(float(np.std(f1s)), 3)}))

    out_csv = REPO_ROOT / "results" / ("pipeline_comparison.csv" if args.stem == "real_grants" else f"pipeline_comparison_{args.stem}.csv")
    out_csv.parent.mkdir(exist_ok=True)
    cols = list(dict.fromkeys(k for row in table for k in row))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print(f"\nWrote {out_csv.relative_to(REPO_ROOT)}")

    if checklist_runs:
        name, rows = next(((n, r) for n, r in checklist_runs if n == args.errors_for), checklist_runs[0])
        print(f"Wrote {write_errors(name, rows, data).relative_to(REPO_ROOT)}")
        if args.stem == "real_grants":
            print(f"Wrote {write_eligibility_checklist(rows, data).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
