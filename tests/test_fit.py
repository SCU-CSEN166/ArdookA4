import pytest

from ardook_grants.fit import entity_rating, explain, fit_label, theme_rating
from ardook_grants.profile import load_profile
from ardook_grants.rules import Verdict


@pytest.fixture(scope="module")
def profile():
    return load_profile()


@pytest.mark.parametrize(
    "theme, entity, fit",
    [
        ("good", "good", "High"),
        ("good", "partial", "Moderate"),
        ("partial", "good", "Moderate"),
        ("partial", "partial", "Moderate"),
        ("bad", "good", "Low"),
        ("good", "bad", "Low"),
    ],
)
def test_team_fit_rule(theme, entity, fit):
    assert fit_label(theme, entity) == fit


def test_theme_takes_the_best_rating(profile):
    rating, areas = theme_rating(["education and scholarships", "disability inclusion"], profile)
    assert rating == "good"
    assert areas["bad"] == ["education and scholarships"]


def test_theme_unknown_areas_are_ignored(profile):
    assert theme_rating(["underwater basket weaving"], profile)[0] is None
    assert theme_rating([], profile)[0] is None


def test_entity_ignores_location():
    v = Verdict("Not a lead", reasons=["applicants limited to United States"],
                geo_reasons=["applicants limited to United States"])
    assert entity_rating(v) == "good"


def test_entity_rating():
    assert entity_rating(Verdict("Not a lead", reasons=["only student applicants"])) == "bad"
    assert entity_rating(Verdict("Conditional lead", conditions=["if Ardook is a registered nonprofit"])) == "partial"


def test_explanation_names_areas_and_conditions(profile):
    v = Verdict("Conditional lead", conditions=["if Ardook is a registered nonprofit"])
    _, areas = theme_rating(["disability inclusion"], profile)
    text = explain("Moderate", "good", areas, "partial", v)
    assert text.startswith("Moderate:") and "disability inclusion" in text and "registered nonprofit" in text
