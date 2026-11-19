"""Make a labeled synthetic grant set for Ardook Data, with the label fixed before the text is written.

Each grant is a set of facts drawn from the pools below (what it funds, who can apply, where, stage limits,
women-founder rules, award). The gold label follows from those facts and docs/ardook-data-profile.md:

    any hard mismatch (nonprofit/university/student/government only, a place Ardook Data doesn't work,
    a revenue or operating-history minimum a new pre-revenue venture can't meet)    -> Low (not relevant)
    otherwise: core theme -> High, adjacent theme -> Moderate, off-mission theme -> Low

An LLM only writes the listing around the facts (in the Grant Tracker's five input columns), so the label
doesn't depend on the model's judgment. Each listing is checked for the words of its deciding fact; listings
that dropped it are discarded. All funders are fictional.

Usage (from the repo root; needs a running OpenAI-compatible server, e.g. deploy/akash/akash_llm.py):
  python src/make_synthetic_ardook_data.py --n 500 --base-url $AKASH_LLM_URL --api-key-env AKASH_LLM_KEY
Writes data/synthetic_ardook_data.csv (git-ignored, like the rest of data/) with columns:
  text, label, relevant, theme, applicant, geo, stage, women, decisive, split   (split: pool / test)
"""

import argparse
import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ardook_grants.backends import APIBackend
from ardook_grants.profile import REPO_ROOT

# (description used in the prompt, words that must appear in the listing)
THEMES = {
    "core": [
        ("AI and machine-learning startups building commercial products", r"\bAI\b|artificial intelligence|machine.learning"),
        ("data and analytics software companies", r"data|analytics"),
        ("early-stage technology startups building software or SaaS products", r"software|SaaS|tech"),
        ("women-founded technology startups", r"wom[ae]n|female"),
        ("women entrepreneurs building a business in any industry", r"wom[ae]n|female"),
        ("startups applying AI to social or economic problems", r"\bAI\b|artificial intelligence"),
    ],
    "adjacent": [
        ("main-street small businesses that want to grow (retail, food, services)", r"small business|business"),
        ("social enterprises in any sector that create jobs", r"social|enterprise|jobs"),
        ("civic engagement and local democracy projects", r"civic|democra|communit"),
        ("digital-skills training programs for underserved communities", r"digital|skills|training"),
        ("community development projects in low-income neighborhoods", r"communit"),
    ],
    "off": [
        ("distributing proven poverty interventions (bed nets, water filters) in low-income countries", r"poverty|distribut"),
        ("community health clinics and medical care", r"health|clinic|medical"),
        ("arts and cultural festivals", r"\barts?\b|cultur|festival"),
        ("wildlife and forest conservation", r"conservation|wildlife|forest"),
        ("tuition scholarships for undergraduate students", r"scholarship|tuition"),
        ("farming cooperatives and agricultural extension", r"farm|agricultur"),
        ("faith-based community programs", r"faith|church|congregation"),
    ],
}
APPLICANTS = {
    "ok": [
        ("for-profit startups", None),
        ("registered companies of any legal form", None),
        ("for-profit or nonprofit organizations", None),
        ("founders and early-stage companies", None),
        ("women-owned small businesses", None),
    ],
    "bad": [
        ("registered 501(c)(3) nonprofit organizations only", r"501|nonprofit|non-profit"),
        ("registered charities and NGOs only (no companies)", r"charit|NGO|non-?governmental|nonprofit"),
        ("accredited universities and research institutes only", r"universit|research institut|academic"),
        ("individual undergraduate students only", r"student|undergraduate"),
        ("government agencies and municipalities only", r"government|municipal"),
    ],
}
PLACES = {
    "ok": [
        ("open worldwide", None),
        ("open to companies based in the United States", r"United States|U\.S\.|USA|\bUS\b"),
        ("open to startups based in California", r"California"),
        ("open to companies in the San Francisco Bay Area", r"Bay Area|San Francisco"),
        ("no geographic restriction (say nothing about location)", None),
    ],
    "bad": [
        ("open only to businesses registered in Kenya", r"Kenya"),
        ("limited to companies based in Texas", r"Texas"),
        ("only for organizations registered in India", r"India"),
        ("limited to businesses in Ontario, Canada", r"Ontario"),
        ("open only to UK-registered companies", r"United Kingdom|\bUK\b|U\.K\."),
        ("limited to businesses located in New York City", r"New York"),
    ],
}
STAGES = {
    "ok": [
        ("no stage requirement (say nothing about company age or revenue)", None),
        ("companies founded within the last 5 years", r"5 years|five years"),
        ("pre-revenue and early-stage startups are welcome", r"pre-revenue|early.stage"),
        ("applicants need at least a prototype or MVP", r"prototype|MVP|minimum viable"),
    ],
    "bad": [
        ("at least 3 years of operating history required", r"3 years|three years"),
        ("minimum annual revenue of $250,000", r"250,000|250k|250K"),
        ("at least $75,000 in annual revenue", r"75,000|75k|75K"),
        ("must have at least 10 full-time employees", r"10 |ten "),
    ],
}
# A second, disjoint set of facts (no phrase shared with the pools above). Grants made from these test whether the
# judge generalizes, or only copies labels from retrieved examples built from the same phrases (--facets alt).
ALT_THEMES = {
    "core": [
        ("computer-vision and robotics startups", r"vision|robot"),
        ("fintech and data-infrastructure ventures", r"fintech|financial|data"),
        ("female-founded consumer startups in any industry", r"female|wom[ae]n"),
        ("generative-AI tools for small businesses", r"generative|\bAI\b"),
        ("women-led e-commerce brands", r"wom[ae]n|female"),
    ],
    "adjacent": [
        ("cooperatives and employee-owned businesses", r"cooperative|co-op|employee-owned"),
        ("rural broadband and connectivity projects", r"broadband|connectivity|internet"),
        ("local tourism and hospitality businesses", r"touris|hospitality"),
        ("youth employment partnerships between companies and schools", r"youth|employment"),
    ],
    "off": [
        ("maternal health programs", r"maternal|health"),
        ("museum exhibitions", r"museum|exhibit"),
        ("coral reef restoration", r"coral|reef"),
        ("K-12 teacher training", r"teacher|K-12"),
        ("refugee legal aid", r"refugee|legal"),
        ("youth sports leagues", r"sport"),
    ],
}
ALT_APPLICANTS = {
    "ok": [("any startup or small company", None), ("sole proprietors and LLCs", None),
           ("privately held for-profit companies", None)],
    "bad": [("public schools and school districts only", r"school"),
            ("hospitals and health systems only", r"hospital|health system"),
            ("tribal governments only", r"tribal"),
            ("faith-based organizations only", r"faith")],
}
ALT_PLACES = {
    "ok": [("open to applicants in any country", None), ("open to U.S.-based companies nationwide", r"U\.S\.|United States|USA|\bUS\b"),
           ("open to companies in Northern California", r"California"), ("applicants from San Francisco or Oakland", r"San Francisco|Oakland")],
    "bad": [("limited to companies in Brazil", r"Brazil"), ("only for firms registered in Germany", r"German"),
            ("limited to Florida businesses", r"Florida"), ("open only to Nigerian startups", r"Nigeria"),
            ("for Australian companies only", r"Australia"), ("limited to Chicago-based businesses", r"Chicago")],
}
ALT_STAGES = {
    "ok": [("companies under 3 years old", r"3 years|three years"), ("idea-stage founders welcome", r"idea"),
           ("no minimum revenue", r"revenue")],
    "bad": [("at least 5 years in business", r"5 years|five years"), ("minimum $1 million annual revenue", r"1 million|\$1M|1,000,000"),
            ("must have raised at least $2 million in venture funding", r"2 million|\$2M|2,000,000"),
            ("minimum of 25 employees", r"25")],
}

WOMEN = [None, "women-led companies only", "at least 51% women-owned", "must have a female founder or co-founder"]
AWARDS = ["$10,000 grant", "$25,000 non-dilutive grant", "$50,000 prize", "$100,000 in cloud credits",
          "accelerator place with a $20,000 stipend", "€60,000 grant", "$5,000 microgrant", "$15,000 cash award"]
TYPES = {"grant": "Grant", "prize": "Prize", "credits": "Cloud credits", "accelerator": "Accelerator", "microgrant": "Microgrant",
         "award": "Award"}
STYLES = ["terse, like a spreadsheet row", "a short funder announcement", "formal call-for-proposals wording",
          "plain bullet-like phrases separated by semicolons"]

SYSTEM = """You write realistic but fictional grant listings for a test dataset.
Write exactly these five lines and nothing else:
Program: <a fictional program and funder name; never a real organization>
Type: <type>
Amount / Benefit: <award>
Deadline: <a date in 2027 as YYYY-MM-DD, or "Rolling">
Core Requirements: <one paragraph>

Core Requirements must state every required fact you are given, in your own words, and must not add any
other eligibility restriction (no extra location, applicant type, age, revenue or identity limits)."""


def sample_facts(rng: random.Random) -> dict:
    """One grant's facts. About 30% get exactly one hard mismatch; themes lean toward core/adjacent."""
    bad_slot = rng.choice(["applicant", "geo", "stage"]) if rng.random() < 0.30 else None
    theme_kind = rng.choices(["core", "adjacent", "off"], weights=[0.40, 0.35, 0.25])[0]
    facts = {
        "theme_kind": theme_kind,
        "theme": rng.choice(THEMES[theme_kind]),
        "applicant": rng.choice(APPLICANTS["bad" if bad_slot == "applicant" else "ok"]),
        "geo": rng.choice(PLACES["bad" if bad_slot == "geo" else "ok"]),
        "stage": rng.choice(STAGES["bad" if bad_slot == "stage" else "ok"]),
        "women": rng.choice(WOMEN),
        "award": rng.choice(AWARDS),
        "style": rng.choice(STYLES),
        "bad_slot": bad_slot,
    }
    return facts


def gold(facts: dict) -> tuple[str, str]:
    """(label, the deciding fact)."""
    if facts["bad_slot"]:
        return "Low", f"{facts['bad_slot']}: {facts[facts['bad_slot']][0]}"
    label = {"core": "High", "adjacent": "Moderate", "off": "Low"}[facts["theme_kind"]]
    return label, f"theme ({facts['theme_kind']}): {facts['theme'][0]}"


def prompt(facts: dict) -> str:
    award = facts["award"]
    kind = next((t for k, t in TYPES.items() if k in award), "Grant")
    lines = [f"Type: {kind}", f"Award: {award}", f"What it funds: {facts['theme'][0]}",
             f"Who can apply: {facts['applicant'][0]}", f"Location: {facts['geo'][0]}", f"Stage: {facts['stage'][0]}"]
    if facts["women"]:
        lines.append(f"Founder rule: {facts['women']}")
    lines.append(f"Writing style for Core Requirements: {facts['style']}")
    return "Required facts:\n" + "\n".join(f"- {line}" for line in lines)


def valid(text: str, facts: dict) -> bool:
    """The five lines are there, the deciding words appear, and the model didn't name the test subject."""
    if not all(re.search(rf"^{f}:", text, re.M) for f in ("Program", "Type", "Amount / Benefit", "Core Requirements")):
        return False
    if re.search(r"ardook", text, re.I):
        return False
    must = [facts["theme"][1]] + [facts[s][1] for s in ("applicant", "geo", "stage") if facts[s][1] and facts["bad_slot"] == s]
    return all(re.search(p, text, re.I) for p in must)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=166)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="AKASH_LLM_KEY")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "synthetic_ardook_data.csv")
    parser.add_argument("--facets", choices=["main", "alt"], default="main",
                        help="alt: the disjoint fact pools; every grant goes to the test split")
    args = parser.parse_args()
    if args.facets == "alt":  # swap the pools that sample_facts() reads
        globals().update(THEMES=ALT_THEMES, APPLICANTS=ALT_APPLICANTS, PLACES=ALT_PLACES, STAGES=ALT_STAGES)

    rng = random.Random(args.seed)
    backend = APIBackend(args.model, args.base_url, args.api_key_env, no_thinking=True)
    specs = [sample_facts(rng) for _ in range(int(args.n * 1.15))]  # a few are discarded by valid()

    def write(i_facts):
        i, facts = i_facts
        text = backend.generate(SYSTEM, prompt(facts), max_new_tokens=350, temperature=0.9, seed=args.seed + i)
        text = "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())
        return facts, text

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        made = list(pool.map(write, enumerate(specs)))

    rows, dropped = [], 0
    seen = set()
    for facts, text in made:
        if not valid(text, facts) or text in seen:
            dropped += 1
            continue
        seen.add(text)
        label, decisive = gold(facts)
        rows.append({
            "text": text, "label": label, "relevant": "Relevant" if label != "Low" else "Not relevant",
            "theme": facts["theme_kind"], "applicant": facts["applicant"][0], "geo": facts["geo"][0],
            "stage": facts["stage"][0], "women": facts["women"] or "", "decisive": decisive,
        })
        if len(rows) == args.n:
            break
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        # pool = retrieval examples, test = scoring only; the alt set is test only
        row["split"] = "test" if args.facets == "alt" or i >= len(rows) // 2 else "pool"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {s: {lab: sum(r["label"] == lab and r["split"] == s for r in rows) for lab in ("High", "Moderate", "Low")}
              for s in ("pool", "test")}
    print(f"Wrote {len(rows)} grants to {args.out} ({dropped} discarded); labels: {json.dumps(counts)}")


if __name__ == "__main__":
    main()
