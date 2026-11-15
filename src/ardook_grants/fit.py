"""Fit: how well a grant suits Ardook (High / Moderate / Low), separate from whether Ardook can apply.

Same rule the team's synthetic data uses (notebooks/grant_fit_iterations.ipynb):
    theme rating   good / partial / bad: what the grant funds, rated in the profile
    entity rating  good / partial / bad: Ardook vs. the grant's organization requirements
    fit            any bad -> Low; both good -> High; otherwise Moderate
Location never changes Fit. It only affects eligibility, which is how Timothy labeled Fit.
"""

from .rules import Verdict

RATINGS = ("bad", "partial", "good")


def theme_rating(focus_areas: list[str], profile: dict) -> tuple[str | None, dict[str, list[str]]]:
    """Best rating among the grant's focus areas (good > partial > bad), plus the areas per rating.

    Taking the best one leans toward keeping grants: missing a good grant is the worst mistake.
    Returns (None, {}) when no focus area was found, so the caller can flag it for review.
    """
    lookup = {c.lower(): r for r in RATINGS for c in profile["theme_categories"][r]}
    by_rating: dict[str, list[str]] = {r: [] for r in RATINGS}
    for area in focus_areas or []:
        rating = lookup.get(area.lower())
        if rating:
            by_rating[rating].append(area)
    found = [r for r in RATINGS if by_rating[r]]
    return (max(found, key=RATINGS.index) if found else None), by_rating


def entity_rating(v: Verdict) -> str:
    """Organization requirements only: a confirmed mismatch is bad, an unknown fact is partial."""
    if v.org_reasons:
        return "bad"
    if v.org_conditions:
        return "partial"
    return "good"


def fit_label(theme: str, entity: str) -> str:
    if "bad" in (theme, entity):
        return "Low"
    if theme == entity == "good":
        return "High"
    return "Moderate"


def explain(fit: str, theme: str | None, areas: dict[str, list[str]], entity: str, v: Verdict) -> str:
    """One plain sentence built only from checked answers, so it can't invent requirements."""
    if theme == "good":
        why = f"matches Ardook's mission ({', '.join(areas['good'])})"
    elif theme == "partial":
        why = f"related to Ardook's work ({', '.join(areas['partial'])})"
    elif theme == "bad":
        why = f"off-mission focus ({', '.join(areas['bad'])})"
    else:
        why = "focus unclear"
    if entity == "bad":
        why += f"; but {'; '.join(v.org_reasons)}"
    elif entity == "partial":
        why += f"; depends on: {'; '.join(v.org_conditions)}"
    return f"{fit}: {why}"
