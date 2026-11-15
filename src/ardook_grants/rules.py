"""Rule-based screen: compare extracted grant requirements with Ardook's profile.

Follows the profile's lead rules: exclude only when a confirmed fact breaks a requirement;
when the deciding fact is unknown, keep the grant as a conditional lead and name the fact.

Every reason and condition is also tagged as either a location or an organization issue. Fit
(see fit.py) uses only the organization ones, since location doesn't change how well a grant suits Ardook.
"""

import re
from dataclasses import dataclass, field
from datetime import date

GLOBAL_TERMS = ("worldwide", "global", "international", "any country", "all countries")
# Wording that names a region rather than one country; Armenia may or may not be included.
REGION_WORDS = ("europe", "eu", "region", "countries", "associated", "asia", "emerging", "developing", "middle-income")
# Funding that isn't a cash grant or prize; whether Ardook wants these is profile Open item 6.
NON_GRANT_FUNDING = {"loan", "equity investment", "in-kind or credits"}
EARLY_STAGES = {"idea or pre-seed", "seed"}


@dataclass
class Verdict:
    status: str  # "Lead", "Conditional lead" or "Not a lead"
    reasons: list[str] = field(default_factory=list)  # confirmed facts that exclude the grant
    conditions: list[str] = field(default_factory=list)  # unknown facts to check
    matches: list[str] = field(default_factory=list)  # requirements Ardook meets
    notes: list[str] = field(default_factory=list)  # already applied, closed, extraction errors
    geo_reasons: list[str] = field(default_factory=list)  # the reasons that are about location
    geo_conditions: list[str] = field(default_factory=list)  # the conditions that are about location

    @property
    def org_reasons(self) -> list[str]:
        return [r for r in self.reasons if r not in self.geo_reasons]

    @property
    def org_conditions(self) -> list[str]:
        return [c for c in self.conditions if c not in self.geo_conditions]

    def update_status(self) -> None:
        self.status = "Not a lead" if self.reasons else "Conditional lead" if self.conditions else "Lead"


def _norm(s: str) -> str:
    return s.lower().replace("’", "'").strip()


# Other ways grants write a country, so "U.S. citizens only" matches a profile listing "United States".
COUNTRY_ALIASES = {
    r"\bu\.?s\.?a?\.?(?=\W|$)": "united states",
    r"\bamerica\b": "united states",
    r"\bunited states of america\b": "united states",
    r"\buk\b|\bu\.k\.": "united kingdom",
}


def _canonical_place(s: str) -> str:
    s = _norm(s)
    for pattern, name in COUNTRY_ALIASES.items():
        s = re.sub(pattern, name, s)
    return s


def classify_location(location: str, profile: dict) -> str:
    """Return 'eligible', 'planned', 'unclear' or 'other' for one extracted location."""
    s = _canonical_place(location)
    if any(_canonical_place(c) in s for c in profile["operating_countries"]):
        return "eligible"
    if any(t in s for t in GLOBAL_TERMS) or any(_norm(r) in s for r in profile.get("eligible_regions", [])):
        return "eligible"
    if any(_canonical_place(c) in s for c in profile.get("planned_countries", [])):
        return "planned"
    if any(re.search(rf"\b{w}\b", s) for w in REGION_WORDS):
        return "unclear"
    return "other"


def _screen_locations(locations: list, profile: dict, kind: str, v: Verdict) -> None:
    if not locations:
        return
    classes = {loc: classify_location(loc, profile) for loc in locations}
    found = set(classes.values())
    listed = ", ".join(locations)
    if "eligible" in found:
        v.matches.append(f"{kind} includes Armenia ({listed})")
    elif "unclear" in found:
        unclear = ", ".join(loc for loc, c in classes.items() if c == "unclear")
        v.geo_conditions.append(f"if Armenia counts under '{unclear}'")
    elif "planned" in found:
        v.geo_conditions.append("if Ardook has launched in Tbilisi")
    else:
        v.geo_reasons.append(f"{kind} limited to {listed}")


MULTI_COUNTRY = (r"\b(europe|european union|eu|asia|africa|latin america|middle east|caucasus|"
                 r"eastern partnership|countries|continent|worldwide|global)\b")


def _place_match(region: str, places: list) -> bool:
    r = _canonical_place(region)
    return any(re.search(rf"\b{re.escape(_canonical_place(p))}\b", r) or _canonical_place(p) in r or r in _canonical_place(p)
               for p in places)


def _screen_regions(regions: list, profile: dict, name: str, v: Verdict) -> None:
    """Grants limited to part of a country (a province, state or town), checked against where the org works."""
    countries = {_canonical_place(c) for c in [*profile.get("operating_countries", []), *profile.get("planned_countries", [])]}
    # Keep only parts of a country: drop answers that are a whole country ("Armenia", "U.S."),
    # several countries ("Europe", "developing countries") or "worldwide".
    regions = [r for r in regions if _canonical_place(r).strip(" .") not in countries
               and not any(t in _norm(r) for t in GLOBAL_TERMS)
               and not re.search(MULTI_COUNTRY, _norm(r))]
    if not regions:
        return
    listed = ", ".join(regions)
    if any(_place_match(r, profile.get("operating_places") or []) for r in regions):
        v.matches.append(f"region includes where {name} works ({listed})")
    elif any(_place_match(r, profile.get("planned_places") or []) for r in regions):
        v.geo_conditions.append(f"if {name} has launched in {listed}")
    elif not profile.get("operating_places"):
        v.geo_conditions.append(f"if {name} works in {listed}")
    else:
        v.geo_reasons.append(f"limited to {listed}")


def location_result(v: Verdict) -> str:
    """'excluded', 'conditional', 'ok' or 'not stated': the location part of the verdict on its own."""
    if v.geo_reasons:
        return "excluded"
    if v.geo_conditions:
        return "conditional"
    return "ok" if any(m.startswith(("applicants includes", "impact includes")) for m in v.matches) else "not stated"


def screen(program: str, ex: dict, profile: dict, today: date | None = None, use_geography: bool = True) -> Verdict:
    today = today or date.today()
    name = profile.get("name", "Ardook")
    v = Verdict(status="Lead")

    if any(_norm(name) in _norm(program) for name in profile.get("already_applied", [])):
        v.notes.append("already applied")
    if "_error" in ex:
        v.notes.append(ex["_error"])
        v.conditions.append("extraction failed; read the grant manually")
        v.status = "Conditional lead"
        return v

    if use_geography:
        _screen_locations(ex.get("applicant_countries") or [], profile, "applicants", v)
        _screen_locations(ex.get("impact_countries") or [], profile, "impact", v)
        if not v.geo_reasons:  # a country mismatch already decides it
            _screen_regions(ex.get("applicant_regions") or [], profile, name, v)
        v.reasons += v.geo_reasons
        v.conditions += v.geo_conditions

    types = set(ex.get("applicant_types") or [])
    not_eligible_as = set(profile.get("not_eligible_as", []))
    if types and types <= not_eligible_as:
        v.reasons.append(f"only {' / '.join(sorted(types))} applicants")
    elif types and not types & {"individual", "any organization"}:
        org_types = types - not_eligible_as
        legal_form = profile.get("legal_form")
        for kind in ("nonprofit", "for-profit company"):
            if org_types != {kind}:
                continue
            if legal_form is None:
                v.conditions.append(f"if {name} is a {'registered nonprofit' if kind == 'nonprofit' else kind}")
            elif legal_form == kind:
                v.matches.append(f"{kind} applicants")
            else:
                v.reasons.append(f"only {kind} applicants; {name} is a {legal_form}")

    if ex.get("women_led_required") == "yes":
        if profile.get("woman_led"):
            v.matches.append("women-led")
        else:
            v.reasons.append("requires a women-led organization")

    share = ex.get("min_women_ownership_percent")
    if share and profile.get("women_ownership_percent") is None:
        v.conditions.append(f"if women own at least {share:g}%")

    if ex.get("program_focus_is_ai_or_deep_tech") == "yes" and profile.get("builds_ai") is False:
        v.reasons.append(f"AI / deep-tech program; {name} doesn't build AI")

    max_age = ex.get("max_company_age_years")
    if max_age and (profile.get("launched_year") is None or profile.get("established_year") is None):
        if profile.get("stage") != "new":  # a brand-new entity meets any age limit
            v.conditions.append(f"if {name} is under {max_age:g} years old")
    elif max_age:
        age_launched = today.year - profile["launched_year"]
        age_established = today.year - profile["established_year"]
        if age_launched > max_age:
            v.reasons.append(f"must be under {max_age:g} years old; {name} launched in {profile['launched_year']}")
        elif age_established > max_age:
            v.conditions.append(f"if company age counts from the {profile['launched_year']} launch (limit {max_age:g} years)")

    min_years = ex.get("min_years_in_operation")
    if min_years:
        if profile.get("established_year") is None or profile.get("launched_year") is None:
            v.conditions.append(f"if {name} has operated at least {min_years:g} years")
        else:
            age_launched = today.year - profile["launched_year"]
            age_established = today.year - profile["established_year"]
            if min_years > age_established:
                v.reasons.append(f"needs {min_years:g}+ years in operation; {name} started in {profile['established_year']}")
            elif min_years > age_launched:
                v.conditions.append(f"if operating years count from {profile['established_year']} (needs {min_years:g}+)")

    min_revenue = ex.get("min_annual_revenue_usd")
    if min_revenue and profile.get("annual_revenue_usd") is None:
        v.conditions.append(f"if annual revenue is at least ${min_revenue:,.0f}")
    elif min_revenue and profile["annual_revenue_usd"] < min_revenue:
        v.reasons.append(f"needs at least ${min_revenue:,.0f} annual revenue")

    max_revenue = ex.get("max_annual_revenue_usd")
    if max_revenue and profile.get("annual_revenue_usd") is None:
        v.conditions.append(f"if annual revenue is under ${max_revenue:,.0f}")

    funding = ex.get("funding_types") or []
    if funding and set(funding) <= NON_GRANT_FUNDING:
        accepted = profile.get("accepted_funding_types")
        if accepted is None:
            v.conditions.append(f"if {name} would take {' / '.join(funding)}")
        elif not set(funding) & set(accepted):
            v.reasons.append(f"funding is {' / '.join(funding)}, which {name} doesn't take")

    stages = ex.get("stages_allowed") or []
    if stages and set(stages) <= EARLY_STAGES and profile.get("stage") == "operating":
        v.conditions.append(f"if an operating, revenue-earning venture still counts as {' / '.join(stages)}")

    identities = ex.get("founder_identity_required") or []
    if identities and profile.get("founder_identity") is None:
        v.conditions.append(f"if the founder is {' / '.join(identities)}")

    v.update_status()
    return v
