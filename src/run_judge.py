"""Rate grants' Fit with the retrieval-augmented judge (src/ardook_grants/judge.py).

Usage (from the repo root; needs an OpenAI-compatible server, e.g. deploy/akash/akash_llm.py):
  python src/run_judge.py --input data/real_grants.csv --labeled-only --profile ardook-data --k 4 --votes 5 \
      --base-url $AKASH_LLM_URL --api-key-env AKASH_LLM_KEY
  python src/run_judge.py --input data/synthetic_ardook_data.csv --split test --profile ardook-data --k 4
  python src/run_judge.py ... --k 0 --votes 1          # no examples: the improved prompt on its own
  python src/run_judge.py ... --rules-only             # re-score from cached answers, no server

Writes outputs/<input>_judge_k<k>_v<votes>[...]/results.csv (same columns as run_pipeline.py's, so it works as
`evaluation/hidden_test.py --fit-run`). The example pool is the sponsor's rated rows for the profile plus the
synthetic pool split; a grant is never its own example.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ardook_grants.backends import AnswerCache, APIBackend, default_api_endpoint, slug
from ardook_grants.data import load_any
import csv

from ardook_grants.judge import (SPONSOR_CSV, SYNTHETIC_ARDOOK_DATA, Judge, Retriever, checklist_facts, default_pool,
                                 judge_profile_text, vote)
from ardook_grants.metrics import fit_scores, format_scores
from ardook_grants.pipeline import process_direct, write_outputs
from ardook_grants.profile import ARDOOK_DATA_PROFILE, PROFILE_PATH, REPO_ROOT, load_profile

PROFILES = {"ardook": PROFILE_PATH, "ardook-data": ARDOOK_DATA_PROFILE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", help="synthetic data only")
    parser.add_argument("--labeled-only", action="store_true")
    parser.add_argument("--profile", default="ardook-data")
    parser.add_argument("--k", type=int, default=16, help="similar rated grants shown as examples (0 = none); 16 was best on held-out data")
    parser.add_argument("--votes", type=int, default=1, help="samples per grant; >1 votes at --temperature")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--facts-run", type=Path, help="a run_pipeline.py output folder on the same input: its "
                        "checklist findings are shown to the judge as hints")
    parser.add_argument("--thinking", action="store_true", help="let Qwen3 reason before answering (slower)")
    parser.add_argument("--pool", choices=["both", "sponsor", "synthetic"], default="both")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--base-url", help="default: AKASH_LLM_URL in .env, else AkashML")
    parser.add_argument("--api-key-env", help="default matches --base-url")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rules-only", action="store_true", help="cached answers only")
    parser.add_argument("--limit", type=int, help="only the first N grants (after --labeled-only)")
    parser.add_argument("--cache", type=Path, help="answer cache file (default: outputs/cache/judge__<model>.json)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    profile_path = PROFILES.get(args.profile, Path(args.profile))
    profile = load_profile(profile_path)
    rows = profile.get("labeled_sheet_rows") or [38, None]
    grants = load_any(args.input, args.split, label_rows=(rows[0], rows[1]))
    grants = [g for g in grants if g.get("gold_fit")] if args.labeled_only else grants
    grants = grants[: args.limit] if args.limit else grants

    sponsor_rows = (rows[0], rows[1] or 10**6)
    pool = default_pool(args.profile, SPONSOR_CSV if args.pool != "synthetic" else Path("-"),
                        sponsor_rows, SYNTHETIC_ARDOOK_DATA if args.pool != "sponsor" else None)
    retriever = Retriever(pool) if args.k > 0 else None
    url, key_env = default_api_endpoint()
    backend = APIBackend(args.model, args.base_url or url, args.api_key_env or key_env, no_thinking=not args.thinking)
    cache = AnswerCache(args.cache or REPO_ROOT / "outputs" / "cache" / f"judge__{slug(args.model)}.json")
    judge = Judge(backend, cache, judge_profile_text(profile_path), retriever, args.k, args.votes, args.temperature,
                  thinking=args.thinking)
    print(f"{len(grants)} grants, pool of {len(pool)} rated examples, k={args.k}, votes={args.votes}")

    facts = {}
    if args.facts_run:
        rows_by_id = {r["id"]: r for r in csv.DictReader((args.facts_run / "results.csv").open(encoding="utf-8"))}
        facts = {g["id"]: checklist_facts(rows_by_id[str(g["id"])]) for g in grants if str(g["id"]) in rows_by_id}
    plans = {g["id"]: judge.requests(g["text"], facts.get(g["id"], "")) for g in grants}
    todo = [(gid, req) for gid, reqs in plans.items() for req in reqs if cache.get("judge", req[0], req[1]) is None]
    if todo and not args.rules_only:
        print(f"Asking {len(todo)} questions with {args.workers} workers ...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool_ex:
            for n, (req, answer) in enumerate(pool_ex.map(lambda t: (t[1], judge.ask(*t[1])), todo), 1):
                cache.put("judge", req[0], req[1], answer, 0.0)
                if n % 50 == 0:
                    cache.save()
        cache.save()

    results = []
    for g in grants:
        hits = [cache.get("judge", v, u) for v, u, _ in plans[g["id"]]]
        if any(h is None for h in hits):
            print(f"skip (not cached): {g['program']}")
            continue
        fit, share, reason = vote([h["answer"] for h in hits])
        r = process_direct(g, {"fit": fit, "reason": reason} if fit else {"_error": "no valid answer"})
        if args.votes > 1:
            r.notes = ", ".join(n for n in (r.notes, f"confidence {share:.2f}", "split vote" if share < 0.6 else "") if n)
            r.needs_review = r.needs_review or share < 0.6
        results.append(r)

    name = f"{args.input.stem}{'_' + args.split if args.split else ''}_judge_k{args.k}_v{args.votes}"
    name += "" if args.pool == "both" else f"_{args.pool}"
    name += "_think" if args.thinking else ""
    name += "_facts" if args.facts_run else ""
    name += "" if args.profile == "ardook" else f"_{slug(Path(args.profile).stem)}"
    out = args.out or REPO_ROOT / "outputs" / name
    write_outputs(results, out)
    labeled = [r for r in results if r.gold_fit]
    if labeled:
        print(format_scores(name, fit_scores([r.gold_fit for r in labeled], [r.fit for r in labeled])))
        rel = [(r.gold_fit != "Low", r.fit != "Low") for r in labeled]
        acc = sum(a == b for a, b in rel) / len(rel)
        print(f"{'':44s} relevant (Fit not Low) accuracy={acc:.3f}")
    print(f"Wrote {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}/results.csv")


if __name__ == "__main__":
    main()
