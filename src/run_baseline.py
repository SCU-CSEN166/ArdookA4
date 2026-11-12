"""Week 6 baseline: NuExtract reads each grant, then the profile's rules screen it.

Usage (from the repo root):
  python src/run_baseline.py                                   # every grant in the spreadsheet
  python src/run_baseline.py --limit 3                         # first 3 grants
  python src/run_baseline.py --input examples/synthetic_grants.json
  python src/run_baseline.py --rules-only                      # re-screen cached extractions, no model

Extractions are cached in outputs/extractions.json so rule changes don't need the model again.
Results go to outputs/baseline_results.csv.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from ardook_grants.data import load_json, load_spreadsheet
from ardook_grants.extract import PROMPT_VERSION
from ardook_grants.profile import REPO_ROOT, load_profile
from ardook_grants.rules import screen

OUT_DIR = REPO_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="JSON file of {program, text} grants (default: spreadsheet)")
    parser.add_argument("--limit", type=int, help="only process the first N grants")
    parser.add_argument("--rules-only", action="store_true", help="use cached extractions; don't load the model")
    args = parser.parse_args()

    grants = load_json(args.input) if args.input else load_spreadsheet()
    grants = grants[: args.limit] if args.limit else grants
    profile = load_profile()

    OUT_DIR.mkdir(exist_ok=True)
    cache_path = OUT_DIR / "extractions.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    extractor = None
    rows = []
    for g in grants:
        cached = cache.get(g["program"])
        if cached and cached["text"] == g["text"] and cached.get("prompt_version") == PROMPT_VERSION:
            extraction, seconds = cached["extraction"], cached["seconds"]
        elif args.rules_only:
            print(f"skip (no cached extraction): {g['program']}")
            continue
        else:
            if extractor is None:
                from ardook_grants.extract import NuExtractor

                print("Loading NuExtract-2.0-2B ...")
                extractor = NuExtractor()
            start = time.perf_counter()
            extraction = extractor.extract(g["text"])
            seconds = round(time.perf_counter() - start, 1)
            cache[g["program"]] = {
                "text": g["text"],
                "prompt_version": PROMPT_VERSION,
                "extraction": extraction,
                "seconds": seconds,
            }
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

        v = screen(g["program"], extraction, profile)
        if g["closed"]:
            v.notes.append("closed")
        rows.append(
            {
                "id": g["id"],
                "program": g["program"],
                "baseline": v.status,
                "why": "; ".join(v.reasons or v.conditions or v.matches or ["no restrictions found"]),
                "notes": ", ".join(v.notes),
                "sheet_fit": g["sheet_fit"],
                "seconds": seconds,
                "extraction": json.dumps(extraction, ensure_ascii=False),
            }
        )
        print(f"[{g['id']:>2}] {v.status:<17} {g['program'][:55]:<55} {seconds:>5}s  {rows[-1]['why']}")

    out_path = OUT_DIR / ("baseline_results.csv" if not args.input else f"{args.input.stem}_results.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    counts = {s: sum(r["baseline"] == s for r in rows) for s in ("Lead", "Conditional lead", "Not a lead")}
    print(f"\n{len(rows)} grants -> {counts}\nWrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
