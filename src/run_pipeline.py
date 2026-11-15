"""Run the full grant pipeline: checklist answers -> eligibility rules -> Fit -> explanation.

Usage (from the repo root):
  python src/run_pipeline.py --input data/real_grants.csv                    # NuExtract-2B on this machine
  python src/run_pipeline.py --input data/real_grants.csv --backend llm      # Qwen/Qwen3-4B (or --model ...)
  python src/run_pipeline.py --input data/real_grants.csv --backend api      # AkashML Llama 3.3 70B ($AKASHML_API_KEY)
  python src/run_pipeline.py --input data/real_grants.csv --rules-only       # reuse cached answers, no model
  python src/run_pipeline.py --input data/real_grants.csv --no-geography     # skip the location rules
  python src/run_pipeline.py --input data/real_grants.csv --backend llm --direct   # baseline: model rates Fit directly
  python src/run_pipeline.py --input data/synthetic_grants.csv --split test --limit 100
  python src/run_pipeline.py --input data/real_grants.csv --labeled-only            # only the 83 rated grants
  python src/run_pipeline.py --input examples/synthetic_grants.json --rules-only --cache examples/cache_nuextract.json
  python src/run_pipeline.py --input hidden.csv --text-columns Program Requirements --label-column Fit

Writes outputs/<input>_<backend>[...]/results.csv, shortlist.csv and trace.jsonl. Model answers are
cached in outputs/cache/, so re-running after a rule change takes seconds. If the input has labels,
the Fit scores are printed at the end (evaluation/evaluate.py compares several runs).
"""

import argparse
from collections import Counter
from pathlib import Path

from ardook_grants.backends import (AKASHML_URL, DEFAULT_API_MODEL, DEFAULT_LLM, NUEXTRACT_MODEL, AnswerCache,
                                    APIBackend, APIError, LLMBackend, NuExtractBackend, slug)
from ardook_grants.checklist import build_groups
from ardook_grants.data import load_any
from ardook_grants.metrics import fit_scores, format_scores
from ardook_grants.pipeline import prefetch, run, write_outputs
from ardook_grants.profile import ARDOOK_DATA_PROFILE, PROFILE_PATH, REPO_ROOT, load_profile, profile_summary

PROFILES = {"ardook": PROFILE_PATH, "ardook-data": ARDOOK_DATA_PROFILE}


def default_out_dir(args, model: str) -> Path:
    name = Path(args.input).stem + (f"_{args.split}" if args.split else "") + f"_{args.backend}"
    if args.backend in ("llm", "api"):
        name += f"_{slug(model.split('/')[-1])}"
    name += "_direct" if args.direct else ""
    name += "_nogeo" if args.no_geography else ""
    name += "" if args.profile == "ardook" else f"_{slug(Path(args.profile).stem)}"
    return REPO_ROOT / "outputs" / name


def print_summary(results) -> None:
    print(f"\n{len(results)} grants")
    print("  Fit:        ", dict(Counter(r.fit for r in results)))
    if results and results[0].eligibility:
        print("  Eligibility:", dict(Counter(r.eligibility for r in results)))
        print("  Location:   ", dict(Counter(r.location for r in results)))
    print("  Needs review:", sum(r.needs_review for r in results))

    labeled = [r for r in results if r.gold_fit]
    if not labeled:
        return
    print("\nFit vs. labels")
    print(format_scores("all labeled grants", fit_scores([r.gold_fit for r in labeled], [r.fit for r in labeled])))
    real = [r for r in labeled if r.synthetic.strip() == "None"]
    if real and len(real) < len(labeled):
        print(format_scores("fully real rows only", fit_scores([r.gold_fit for r in real], [r.fit for r in real])))
    for step in ("theme", "entity"):
        pairs = [(getattr(r, f"gold_{step}"), getattr(r, step)) for r in labeled if getattr(r, f"gold_{step}")]
        if pairs:
            acc = sum(g == p for g, p in pairs) / len(pairs)
            print(f"  {step} rating accuracy (synthetic labels): {acc:.3f}  (n={len(pairs)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="grants: Grant Tracker CSV/XLSX, synthetic CSV, JSON, or any CSV")
    parser.add_argument("--split", help="synthetic data only: train, val or test")
    parser.add_argument("--text-columns", nargs="+", help="generic CSV: columns that make up the grant text")
    parser.add_argument("--label-column", help="generic CSV: column holding the Fit label")
    parser.add_argument("--backend", choices=["nuextract", "llm", "api"], default="nuextract",
                        help="nuextract / llm run on this machine; api calls a hosted model (default: AkashML)")
    parser.add_argument("--model", help=f"model name (defaults: {NUEXTRACT_MODEL}, {DEFAULT_LLM}, {DEFAULT_API_MODEL})")
    parser.add_argument("--base-url", default=AKASHML_URL, help="api only: OpenAI-compatible base URL")
    parser.add_argument("--api-key-env", default="AKASHML_API_KEY", help="api only: env var (or .env line) holding the key")
    parser.add_argument("--no-thinking", action="store_true", help="api only: turn off Qwen3 thinking (vLLM servers)")
    parser.add_argument("--workers", type=int, default=1, help="api only: questions asked in parallel")
    parser.add_argument("--rules-only", action="store_true", help="use cached answers only; don't load a model")
    parser.add_argument("--no-geography", action="store_true", help="turn off the location rules")
    parser.add_argument("--direct", action="store_true", help="llm only: baseline that asks the model for Fit directly")
    parser.add_argument("--limit", type=int, help="only the first N grants")
    parser.add_argument("--labeled-only", action="store_true", help="only grants with a Fit label")
    parser.add_argument("--profile", default="ardook", help="ardook, ardook-data, or a path to a profile .md")
    parser.add_argument("--cache", type=Path, help="answer cache file (default: outputs/cache/<backend>__<model>.json)")
    parser.add_argument("--out", type=Path, help="output folder")
    parser.add_argument("--quiet", action="store_true", help="don't print a line per grant")
    args = parser.parse_args()
    if args.direct and args.backend == "nuextract":
        parser.error("--direct needs --backend llm or api")

    model = args.model or {"llm": DEFAULT_LLM, "api": DEFAULT_API_MODEL}.get(args.backend, NUEXTRACT_MODEL)
    if args.backend == "api":
        backend = APIBackend(model, args.base_url, args.api_key_env, no_thinking=args.no_thinking)
    else:
        backend = LLMBackend(model) if args.backend == "llm" else NuExtractBackend(model)
    cache = AnswerCache(args.cache or REPO_ROOT / "outputs" / "cache" / f"{args.backend}__{slug(model)}.json")

    profile_path = PROFILES.get(args.profile, Path(args.profile))
    profile = load_profile(profile_path)
    rows = profile.get("labeled_sheet_rows") or [38, None]
    grants = load_any(args.input, args.split, args.text_columns, args.label_column, (rows[0], rows[1]))
    grants = [g for g in grants if g.get("gold_fit")] if args.labeled_only else grants
    grants = grants[: args.limit] if args.limit else grants
    groups = build_groups(profile)
    try:
        if args.backend == "api" and args.workers > 1 and not args.rules_only:
            asked = prefetch(grants, groups, backend, cache, args.workers, direct=args.direct,
                             profile_text=profile_summary(profile_path) if args.direct else "")
            print(f"Asked {asked} new questions with {args.workers} workers")
        results = run(
            grants, groups, profile, backend, cache,
            rules_only=args.rules_only, use_geography=not args.no_geography, direct=args.direct,
            profile_text=profile_summary(profile_path) if args.direct else "", verbose=not args.quiet,
        )
    except APIError as e:
        cache.save()
        raise SystemExit(f"Stopped: {e}\nAnswers so far are cached; re-run the same command to continue.")
    if not results:
        print("No grants processed.")
        return
    out_dir = args.out or default_out_dir(args, model)
    write_outputs(results, out_dir)
    print_summary(results)
    shown = out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir
    print(f"\nWrote {shown}/results.csv, shortlist.csv, trace.jsonl")


if __name__ == "__main__":
    main()
