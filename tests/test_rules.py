from datetime import date

import pytest

from ardook_grants.profile import load_profile
from ardook_grants.rules import classify_location, screen

TODAY = date(2026, 11, 3)


@pytest.fixture
def profile():
    # Fixed copy so tests don't break when Ardook answers an Open item in the real profile.
    return {
        "operating_countries": ["Armenia"],
        "planned_countries": ["Georgia"],
        "eligible_regions": ["Eastern Partnership", "South Caucasus"],
        "woman_led": True,
        "women_ownership_percent": None,
        "legal_form": None,
        "established_year": 2020,
        "launched_year": 2022,
        "builds_ai": False,
        "annual_revenue_usd": None,
        "already_applied": ["Tara's Circle"],
    }


def test_real_profile_has_the_fields_rules_use():
    p = load_profile()
    for key in ("operating_countries", "planned_countries", "woman_led", "legal_form",
                "established_year", "launched_year", "builds_ai", "already_applied"):
        assert key in p


@pytest.mark.parametrize(
    "location, expected",
    [
        ("Armenia", "eligible"),
        ("worldwide", "eligible"),
        ("Eastern Partnership countries", "eligible"),
        ("Georgia", "planned"),
        ("EU member states", "unclear"),
        ("United States", "other"),
        ("India", "other"),
    ],
)
def test_classify_location(location, expected, profile):
    assert classify_location(location, profile) == expected


def test_us_only_is_not_a_lead(profile):
    v = screen("X", {"applicant_countries": ["United States"]}, profile, TODAY)
    assert v.status == "Not a lead"


def test_impact_elsewhere_is_not_a_lead(profile):
    v = screen("X", {"applicant_countries": [], "impact_countries": ["India"]}, profile, TODAY)
    assert v.status == "Not a lead"


def test_georgia_only_is_conditional(profile):
    v = screen("X", {"applicant_countries": ["Georgia"]}, profile, TODAY)
    assert v.status == "Conditional lead"
    assert "Tbilisi" in v.conditions[0]


def test_unknown_legal_form_keeps_nonprofit_grant_conditional(profile):
    v = screen("X", {"applicant_countries": ["worldwide"], "applicant_types": ["nonprofit"]}, profile, TODAY)
    assert v.status == "Conditional lead"
    assert "nonprofit" in v.conditions[0]


def test_ai_program_excludes(profile):
    v = screen("X", {"program_focus_is_ai_or_deep_tech": "yes"}, profile, TODAY)
    assert v.status == "Not a lead"


def test_age_limit_depends_on_which_date_counts(profile):
    # 2026: 6 years since established, 4 since launch.
    assert screen("X", {"max_company_age_years": 5}, profile, TODAY).status == "Conditional lead"
    assert screen("X", {"max_company_age_years": 3}, profile, TODAY).status == "Not a lead"
    assert screen("X", {"max_company_age_years": 10}, profile, TODAY).status == "Lead"


def test_nothing_stated_is_a_lead(profile):
    assert screen("X", {"applicant_countries": [], "women_led_required": "yes"}, profile, TODAY).status == "Lead"


def test_already_applied_is_flagged_with_curly_apostrophe(profile):
    assert "already applied" in screen("Tara’s Circle 2026", {}, profile, TODAY).notes


def test_bad_extraction_is_conditional_not_excluded(profile):
    v = screen("X", {"_error": "model output was not valid JSON", "_raw": "{"}, profile, TODAY)
    assert v.status == "Conditional lead"
