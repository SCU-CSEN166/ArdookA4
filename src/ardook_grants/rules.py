"""Rule-based screen: compare extracted grant requirements with Ardook's profile.

Follows the profile's lead rules: exclude only when a confirmed fact breaks a requirement;
when the deciding fact is unknown, keep the grant as a conditional lead and name the fact.
"""

import re
from dataclasses import dataclass, field
from datetime import date

GLOBAL_TERMS = ("worldwide", "global", "international", "any country", "all countries")
# Wording that names a region rather than one country; Armenia may or may not be included.
REGION_WORDS = ("europe", "eu", "region", "countries", "associated", "asia", "emerging", "developing", "middle-income")


@dataclass
class Verdict:
    status: str  # "Lead", "Conditional lead" or "Not a lead"
    reasons: list[str] = field(default_factory=list)  # confirmed facts that exclude the grant
    conditions: list[str] = field(default_factory=list)  # unknown facts to check
    matches: list[str] = field(default_factory=list)  # requirements Ardook meets
    notes: list[str] = field(default_factory=list)  # already applied, closed, extraction errors


def _norm(s: str) -> str:
    return s.lower().replace("’", "'").strip()


def classify_location(location: str, profile: dict) -> str:
    """Return 'eligible', 'planned', 'unclear' or 'other' for one extracted location."""
    s = _norm(location)
    if any(_norm(c) in s for c in profile["operating_countries"]):
        return "eligible"
    if any(t in s for t in GLOBAL_TERMS) or any(_norm(r) in s for r in profile.get("eligible_regions", [])):
        return "eligible"
    if any(_norm(c) in s for c in profile.get("planned_countries", [])):
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
        v.conditions.append(f"if Armenia counts under '{unclear}'")
    elif "planned" in found:
        v.conditions.append("if Ardook has launched in Tbilisi")
    else:
        v.reasons.append(f"{kind} limited to {listed}")


def screen(program: str, ex: dict, profile: dict, today: date | None = None) -> Verdict:
    today = today or date.today()
    v = Verdict(status="Lead")

    if any(_norm(name) in _norm(program) for name in profile.get("already_applied", [])):
        v.notes.append("already applied")
    if "_error" in ex:
        v.notes.append(ex["_error"])
        v.conditions.append("extraction failed; read the grant manually")
        v.status = "Conditional lead"
        return v

    _screen_locations(ex.get("applicant_countries") or [], profile, "applicants", v)
    _screen_locations(ex.get("impact_countries") or [], profile, "impact", v)

    types = set(ex.get("applicant_types") or [])
    if types and "individual" not in types and profile.get("legal_form") is None:
        if types == {"nonprofit"}:
            v.conditions.append("if Ardook is a registered nonprofit")
        elif types == {"for-profit company"}:
            v.conditions.append("if Ardook is a for-profit company")

    if ex.get("women_led_required") == "yes":
        if profile.get("woman_led"):
            v.matches.append("women-led")
        else:
            v.reasons.append("requires a women-led organization")

    share = ex.get("min_women_ownership_percent")
    if share and profile.get("women_ownership_percent") is None:
        v.conditions.append(f"if women own at least {share:g}%")

    if ex.get("program_focus_is_ai_or_deep_tech") == "yes" and profile.get("builds_ai") is False:
        v.reasons.append("AI / deep-tech program; Ardook doesn't build AI")

    max_age = ex.get("max_company_age_years")
    if max_age:
        age_launched = today.year - profile["launched_year"]
        age_established = today.year - profile["established_year"]
        if age_launched > max_age:
            v.reasons.append(f"must be under {max_age} years old; Ardook launched in {profile['launched_year']}")
        elif age_established > max_age:
            v.conditions.append(f"if company age counts from the {profile['launched_year']} launch (limit {max_age} years)")

    max_revenue = ex.get("max_annual_revenue_usd")
    if max_revenue and profile.get("annual_revenue_usd") is None:
        v.conditions.append(f"if annual revenue is under ${max_revenue:,.0f}")

    if v.reasons:
        v.status = "Not a lead"
    elif v.conditions:
        v.status = "Conditional lead"
    return v
