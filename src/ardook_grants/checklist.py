"""The questions the model answers about each grant, plus checks on its answers.

Four small question groups, each asked in its own model call (Week 6 finding: the 2B model
answers one short question set better than one long one):

  applicants    who can apply, where, size/age/ownership limits   (+ supporting quotes)
  ai            is the program about AI / deep tech?              (same question as Week 6)
  requirements  funding type, required stage, founder identity    (+ supporting quotes)
  theme         what the grant funds, from the profile's theme categories (+ supporting quotes)

Templates use NuExtract's notation; the LLM backend gets the same templates explained in words.
After the model answers, `clean()` drops values outside the allowed lists and checks that every
quote really appears in the grant text. A made-up quote voids the answers it was meant to support.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .extract import _BRIGHT_FUTURES, _OPEN_HORIZONS, AI_EXAMPLES, AI_TEMPLATE

APPLICANT_TYPES = [
    "nonprofit",
    "for-profit company",
    "any organization",
    "individual",
    "student",
    "university or school",
    "government agency",
    "hospital or clinic",
]
FUNDING_TYPES = ["grant", "prize", "loan", "equity investment", "in-kind or credits", "fellowship", "accelerator"]
STAGES = ["idea or pre-seed", "seed", "growth"]
FOUNDER_IDENTITIES = ["racial or ethnic minority", "LGBTQ+", "veteran", "immigrant or refugee"]

# What a grant can fund. These describe grants, not an organization, so every profile rates this same
# list (good / partial / bad) and cached theme answers can be re-judged for another organization.
# Order matters: it is part of the question, so changing it re-asks every theme question.
THEME_CATEGORIES = [
    "jobs or income for mothers and caregivers",
    "disability inclusion",
    "women's economic empowerment and decent work",
    "home-based or flexible work",
    "social enterprises that employ women",
    "reducing inequality",
    "women entrepreneurs in any industry",
    "small business growth",
    "gender equality and women's rights",
    "family well-being",
    "poverty reduction and community development",
    "social innovation and social ventures",
    "workforce development and job training",
    "eco-friendly business",
    "social justice and equity",
    "nonprofit capacity building",
    "women's leadership development",
    "education and scholarships",
    "science and research",
    "AI and technology products",
    "health and medicine",
    "violence prevention",
    "politics and policy",
    "arts and culture",
    "environment and conservation",
    "LGBTQ+ programs",
    "disaster relief and emergency response",
    "legal advocacy",
    "sports, religion or infrastructure",
]

QUOTE_THRESHOLD = 0.9  # share of a quote that must match the grant text word for word

# Answers that only count if the grant text actually uses matching words. NuExtract-2B sometimes
# guesses these (e.g. "seed stage" for any small-business grant), and each guess adds a condition.
WORD_GUARDS = {
    "stages_allowed": r"seed|idea stage|early[- ]stage|growth[- ]stage|scale-?up|pre-?revenue",
    "founder_identity_required": r"black|latin|hispanic|minorit|of color|indigenous|native|lgbt|queer|veteran"
                                 r"|immigrant|refugee|ethnic|racial|race|bipoc|asian|arab",
}
# Applicant types that rule a grant out on their own. Small models often answer "nonprofit" for any
# impact-flavored grant (Women TechEU, Tara's Circle, Amber Grant), so the listing must say it.
APPLICANT_TYPE_WORDS = {
    "nonprofit": r"non-?profit|not-for-profit|501\s*\(?c\)?|charit|\bngos?\b|non-?governmental|civil society|\bcsos?\b",
    "student": r"student|undergraduate|graduate|phd|enrolled",
    "university or school": r"universit|college|school|academic|research institut",
    "government agency": r"government|municipal|public agenc|ministr",
    "hospital or clinic": r"hospital|clinic|health (?:system|provider)",
}
NON_GRANT_WORDS = {"loan": r"loan|lend|repay|interest", "equity investment": r"equity|invest|stake|shares",
                   "in-kind or credits": r"credit|in-kind|in kind|services|software|cloud"}

# Synthetic grants (not from the dataset) used as in-context examples.
_HEARTHSIDE = (
    "Program: Hearthside Livelihoods Grant\nType: Grant\nAmount / Benefit: €15,000\nDeadline: 2027-01-31\n"
    "Core Requirements: Open to registered organizations of any legal form in Eastern Partnership countries; "
    "projects must create paid home-based work for mothers caring for children with disabilities"
)
_LUMEN = (
    "Program: Lumen STEM Scholarship\nType: Scholarship\nAmount / Benefit: $5,000\nDeadline: 2026-12-15\n"
    "Core Requirements: Female undergraduate students pursuing degrees in engineering or computer science; "
    "U.S. citizens only"
)
_COPPERLEAF = (
    "Program: Copperleaf Women's Growth Loan\nType: Loan\nAmount / Benefit: Up to $50,000 at 0% interest\n"
    "Deadline: Rolling\nCore Requirements: Women-owned small businesses in any industry; seed or growth stage"
)
_RIVERTON = (
    "Program: Riverton Main Street Award\nType: Prize\nAmount / Benefit: $20,000\nDeadline: 2027-04-01\n"
    "Core Requirements: For-profit businesses in Cook County, Illinois with at least 2 years in operation "
    "and at least $100,000 in annual revenue"
)
_SPARK = (
    "Program: Spark Seed Grant\nType: Seed funding\nAmount / Benefit: $10,000\nDeadline: Rolling\n"
    "Core Requirements: New ventures founded less than 18 months ago that have raised under $50,000; "
    "projects must be based in Shirak or Lori provinces, Armenia"
)
_BLOOM = (
    "Program: Bloom Founders Fund\nType: Equity investment\nAmount / Benefit: $100,000 for 5% equity\n"
    "Deadline: 2027-03-01\nCore Requirements: Pre-seed startups founded by Black or Latina women"
)


def _example(text: str, answer: dict) -> dict:
    return {"input": text, "output": json.dumps(answer, ensure_ascii=False)}


@dataclass(frozen=True)
class Group:
    name: str
    template: dict
    examples: list
    # evidence field -> the answers its quotes back up
    evidence: dict = field(default_factory=dict)

    @property
    def version(self) -> str:
        """Changes whenever the question or its examples change, so cached answers get redone."""
        blob = json.dumps([self.name, self.template, self.examples], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def build_groups(profile: dict) -> dict[str, Group]:
    """The four question groups. The theme categories come from the profile."""
    rated = {c for rating in ("good", "partial", "bad") for c in profile["theme_categories"][rating]}
    if rated != set(THEME_CATEGORIES):
        raise ValueError(f"The profile must rate exactly the categories in checklist.THEME_CATEGORIES; "
                         f"missing {sorted(set(THEME_CATEGORIES) - rated)}, unknown {sorted(rated - set(THEME_CATEGORIES))}")
    categories = THEME_CATEGORIES

    applicants = Group(
        "applicants",
        {
            "applicant_countries": ["string"],
            "impact_countries": ["string"],
            "applicant_types": [APPLICANT_TYPES],
            "women_led_required": ["yes", "no"],
            "min_women_ownership_percent": "number",
            "max_company_age_years": "integer",
            "max_annual_revenue_usd": "number",
            "award": "verbatim-string",
            "deadline": "verbatim-string",
            "applicant_evidence": ["verbatim-string"],
        },
        [
            _example(_BRIGHT_FUTURES, {
                "applicant_countries": ["Canada"], "impact_countries": ["Kenya"],
                "applicant_types": ["nonprofit"], "women_led_required": "yes",
                "min_women_ownership_percent": None, "max_company_age_years": 3,
                "max_annual_revenue_usd": None, "award": "$25,000", "deadline": "2026-03-01",
                "applicant_evidence": ["Registered charity based in Canada",
                                       "projects must serve communities in Kenya",
                                       "women-led organizations only", "organization under 3 years old"],
            }),
            _example(_HEARTHSIDE, {
                "applicant_countries": ["Eastern Partnership countries"], "impact_countries": [],
                "applicant_types": ["any organization"], "women_led_required": None,
                "min_women_ownership_percent": None, "max_company_age_years": None,
                "max_annual_revenue_usd": None, "award": "€15,000", "deadline": "2027-01-31",
                "applicant_evidence": ["Open to registered organizations of any legal form in Eastern Partnership countries"],
            }),
            _example(_LUMEN, {
                "applicant_countries": ["United States"], "impact_countries": [],
                "applicant_types": ["student"], "women_led_required": None,
                "min_women_ownership_percent": None, "max_company_age_years": None,
                "max_annual_revenue_usd": None, "award": "$5,000", "deadline": "2026-12-15",
                "applicant_evidence": ["Female undergraduate students pursuing degrees in engineering or computer science",
                                       "U.S. citizens only"],
            }),
        ],
        {"applicant_evidence": ["applicant_countries", "impact_countries", "applicant_types", "women_led_required",
                                "min_women_ownership_percent", "max_company_age_years", "max_annual_revenue_usd"]},
    )

    ai = Group("ai", AI_TEMPLATE, AI_EXAMPLES)

    requirements = Group(
        "requirements",
        {
            "funding_types": [FUNDING_TYPES],
            "stages_allowed": [STAGES],
            "founder_identity_required": [FOUNDER_IDENTITIES],
            "requirements_evidence": ["verbatim-string"],
        },
        [
            _example(_COPPERLEAF, {
                "funding_types": ["loan"], "stages_allowed": ["seed", "growth"], "founder_identity_required": [],
                "requirements_evidence": ["Type: Loan", "seed or growth stage"],
            }),
            _example(_BLOOM, {
                "funding_types": ["equity investment"], "stages_allowed": ["idea or pre-seed"],
                "founder_identity_required": ["racial or ethnic minority"],
                "requirements_evidence": ["$100,000 for 5% equity", "Pre-seed startups founded by Black or Latina women"],
            }),
            _example(_HEARTHSIDE, {
                "funding_types": ["grant"], "stages_allowed": [], "founder_identity_required": [],
                "requirements_evidence": ["Type: Grant"],
            }),
        ],
        {"requirements_evidence": ["funding_types", "stages_allowed", "founder_identity_required"]},
    )

    theme_examples = [
        (_HEARTHSIDE, ["jobs or income for mothers and caregivers", "home-based or flexible work", "disability inclusion"],
         ["projects must create paid home-based work for mothers caring for children with disabilities"]),
        (_LUMEN, ["education and scholarships"],
         ["Female undergraduate students pursuing degrees in engineering or computer science"]),
        (_COPPERLEAF, ["women entrepreneurs in any industry", "small business growth"],
         ["Women-owned small businesses in any industry"]),
        (_OPEN_HORIZONS, ["AI and technology products"], ["for-profit startups building AI products"]),
    ]
    missing = {c for _, areas, _ in theme_examples for c in areas} - set(categories)
    if missing:
        raise ValueError(f"Theme examples use categories the profile no longer has: {sorted(missing)}")
    theme = Group(
        "theme",
        {"focus_areas": [categories], "focus_evidence": ["verbatim-string"]},
        [_example(t, {"focus_areas": areas, "focus_evidence": quotes}) for t, areas, quotes in theme_examples],
        {"focus_evidence": ["focus_areas"]},
    )
    # Asked separately from "applicants" (and added later): limits the models confused there, such as
    # "at least $75K revenue" read as a cap, "under 18 months" read as 18 years, and provinces
    # flattened to the country. "Raised" money is not revenue.
    limits = Group(
        "limits",
        {
            "applicant_regions": ["string"],
            "min_years_in_operation": "number",
            "max_years_in_operation": "number",
            "min_annual_revenue_usd": "number",
            "max_annual_revenue_usd": "number",
            "limits_evidence": ["verbatim-string"],
        },
        [
            _example(_RIVERTON, {
                "applicant_regions": ["Cook County, Illinois"], "min_years_in_operation": 2,
                "max_years_in_operation": None, "min_annual_revenue_usd": 100000, "max_annual_revenue_usd": None,
                "limits_evidence": ["businesses in Cook County, Illinois with at least 2 years in operation",
                                    "at least $100,000 in annual revenue"],
            }),
            _example(_SPARK, {
                "applicant_regions": ["Shirak", "Lori"], "min_years_in_operation": None,
                "max_years_in_operation": 1.5, "min_annual_revenue_usd": None, "max_annual_revenue_usd": None,
                "limits_evidence": ["founded less than 18 months ago",
                                    "projects must be based in Shirak or Lori provinces, Armenia"],
            }),
            _example(_BRIGHT_FUTURES, {
                "applicant_regions": [], "min_years_in_operation": None, "max_years_in_operation": 3,
                "min_annual_revenue_usd": None, "max_annual_revenue_usd": None,
                "limits_evidence": ["organization under 3 years old"],
            }),
            _example(_COPPERLEAF, {
                "applicant_regions": [], "min_years_in_operation": None, "max_years_in_operation": None,
                "min_annual_revenue_usd": None, "max_annual_revenue_usd": None, "limits_evidence": [],
            }),
        ],
        {"limits_evidence": ["applicant_regions", "min_years_in_operation", "max_years_in_operation",
                             "min_annual_revenue_usd", "max_annual_revenue_usd"]},
    )
    return {g.name: g for g in (applicants, ai, requirements, theme, limits)}


# ---------------------------------------------------------------------------------------------
# Checking answers


def _kind(spec) -> str:
    if isinstance(spec, str):
        return spec  # "string", "verbatim-string", "number", "integer"
    if spec and isinstance(spec[0], list):
        return "multi"
    if spec and spec[0] in ("string", "verbatim-string"):
        return "list"
    return "enum"


def empty_value(spec):
    return [] if _kind(spec) in ("multi", "list") else None


def _norm(s: str) -> str:
    s = str(s).lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-"), (" ", " ")):
        s = s.replace(a, b)
    return " ".join(s.split())


def _to_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    s = _norm(value).replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("%", "").strip()
    m = re.fullmatch(r"(?:usd\s*)?(\d+(?:\.\d+)?)\s*(k|thousand|m|million|mn)?(?:\s*usd)?", s)
    if not m:
        return None
    n = float(m.group(1)) * {None: 1, "k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "mn": 1e6}[m.group(2)]
    return int(n) if n.is_integer() else n


def _match_option(value, options: list[str]):
    lookup = {_norm(o): o for o in options}
    return lookup.get(_norm(value)) if isinstance(value, str) else None


def validate(group: Group, raw: dict) -> tuple[dict, list[str]]:
    """Keep only well-formed answers: allowed values, real numbers, lists where lists are expected."""
    facts, problems = {}, []
    for name, spec in group.template.items():
        value, kind = raw.get(name), _kind(spec)
        if value is None or value == "" or value == []:
            facts[name] = empty_value(spec)
            continue
        if kind == "enum":
            facts[name] = _match_option(value, spec)
        elif kind == "multi":
            items = value if isinstance(value, list) else [value]
            # Chat models sometimes copy the template's [[...]] notation: [["a", "b"]] means ["a", "b"].
            items = [v for item in items for v in (item if isinstance(item, list) else [item])]
            matched = [_match_option(v, spec[0]) for v in items]
            facts[name] = list(dict.fromkeys(m for m in matched if m))
            dropped = [v for v, m in zip(items, matched) if not m]
            if dropped:
                problems.append(f"{name}: dropped {dropped} (not allowed values)")
            continue
        elif kind == "list":
            items = value if isinstance(value, list) else [value]
            facts[name] = list(dict.fromkeys(str(v).strip() for v in items if str(v).strip()))
            continue
        elif kind in ("number", "integer"):
            n = _to_number(value)
            facts[name] = round(n) if (n is not None and kind == "integer") else n
        else:
            facts[name] = str(value).strip() if not isinstance(value, (list, dict)) else None
        if facts[name] is None:
            problems.append(f"{name}: dropped {value!r}")
    return facts, problems


def _loose(s: str) -> str:
    """Lowercase words only: punctuation (hyphens, commas, quotes) doesn't matter when matching quotes."""
    return " ".join(re.sub(r"[^\w\s]", " ", _norm(s)).split())


def quote_found(quote: str, text: str, threshold: float = QUOTE_THRESHOLD) -> bool:
    """True if the quote is in the text, ignoring case and punctuation, allowing small differences
    and '...' between parts."""
    t = _loose(text)
    parts = [_loose(p) for p in re.split(r"\.\.\.|…", _norm(quote))]
    parts = [p for p in parts if len(p) > 3]
    if not parts:
        return True  # nothing checkable, e.g. "..."
    for p in parts:
        if p in t:
            continue
        m = SequenceMatcher(None, p, t, autojunk=False).find_longest_match(0, len(p), 0, len(t))
        if m.size / len(p) < threshold:
            return False
    return True


def apply_word_guards(facts: dict, text: str) -> list[str]:
    """Drop guessed answers whose wording isn't in the grant text. Returns what was dropped."""
    problems, t = [], _norm(text)
    for name, pattern in WORD_GUARDS.items():
        if facts.get(name) and not re.search(pattern, t):
            problems.append(f"{name}: dropped {facts[name]} (no such wording in the grant)")
            facts[name] = []
    types = facts.get("applicant_types") or []
    for kind, pattern in APPLICANT_TYPE_WORDS.items():
        if kind in types and not re.search(pattern, t):
            problems.append(f"applicant_types: dropped {kind} (no such wording in the grant)")
            types = [x for x in types if x != kind]
    if "applicant_types" in facts:
        facts["applicant_types"] = types
    funding = facts.get("funding_types") or []
    kept = [f for f in funding if f not in NON_GRANT_WORDS or re.search(NON_GRANT_WORDS[f], t)]
    if kept != funding:
        problems.append(f"funding_types: dropped {sorted(set(funding) - set(kept))} (no such wording in the grant)")
        facts["funding_types"] = kept
    return problems


def clean(group: Group, raw: dict, text: str) -> tuple[dict, dict]:
    """Validate one group's raw answer and check its quotes.

    Returns (facts, report). report = {"error", "problems", "quotes": [{field, quote, found}],
    "unverified": [fields voided because their quote wasn't in the text]}.
    """
    report = {"error": None, "problems": [], "quotes": [], "unverified": []}
    if not isinstance(raw, dict) or "_error" in raw:
        report["error"] = (raw or {}).get("_error", "no answer") if isinstance(raw, dict) else "no answer"
        return {}, report
    facts, report["problems"] = validate(group, raw)
    report["problems"] += apply_word_guards(facts, text)
    for ev_field, supported in group.evidence.items():
        checks = [{"field": ev_field, "quote": q, "found": quote_found(q, text)} for q in facts.get(ev_field) or []]
        report["quotes"] += checks
        has_answers = any(facts.get(f) not in (None, []) for f in supported)
        if has_answers and any(not c["found"] for c in checks):
            for f in supported:
                if facts.get(f) not in (None, []):
                    report["unverified"].append(f)
                facts[f] = empty_value(group.template[f])
    return facts, report


def parse_json_answer(raw: str) -> dict:
    """Parse a model's JSON answer, tolerating code fences or text around the object."""
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()  # reasoning models
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        value = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        try:
            value = json.loads(s[start : end + 1]) if start != -1 and end > start else None
        except json.JSONDecodeError:
            value = None
    if not isinstance(value, dict):
        return {"_error": "model output was not valid JSON", "_raw": raw}
    return value
