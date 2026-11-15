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


# --- Location tags and organization checks (pipeline v2) -------------------------------------


@pytest.fixture
def full_profile(profile):
    return {**profile, "stage": "operating", "accepted_funding_types": None, "founder_identity": None,
            "not_eligible_as": ["student", "university or school", "government agency", "hospital or clinic"]}


def test_location_reasons_are_tagged(profile):
    v = screen("X", {"applicant_countries": ["United States"], "applicant_types": ["nonprofit"]}, profile, TODAY)
    assert v.geo_reasons == ["applicants limited to United States"]
    assert v.org_reasons == [] and v.org_conditions == ["if Ardook is a registered nonprofit"]


def test_no_geography_skips_only_location(profile):
    v = screen("X", {"applicant_countries": ["United States"], "applicant_types": ["nonprofit"]}, profile, TODAY,
               use_geography=False)
    assert v.status == "Conditional lead" and v.geo_reasons == []


@pytest.mark.parametrize("types", [["student"], ["university or school"], ["student", "government agency"]])
def test_only_non_organization_applicants_excludes(types, full_profile):
    assert screen("X", {"applicant_types": types}, full_profile, TODAY).status == "Not a lead"


def test_individual_founders_are_not_excluded(full_profile):
    # Many women-founder grants are applied for by the founder personally.
    assert screen("X", {"applicant_types": ["individual"]}, full_profile, TODAY).status == "Lead"


def test_any_organization_needs_no_legal_form_check(full_profile):
    assert screen("X", {"applicant_types": ["any organization", "nonprofit"]}, full_profile, TODAY).status == "Lead"


def test_loan_or_equity_is_conditional(full_profile):
    v = screen("X", {"funding_types": ["loan"]}, full_profile, TODAY)
    assert v.status == "Conditional lead" and "loan" in v.conditions[0]
    assert screen("X", {"funding_types": ["loan", "grant"]}, full_profile, TODAY).status == "Lead"


def test_known_funding_preference_excludes(full_profile):
    p = {**full_profile, "accepted_funding_types": ["grant", "prize"]}
    assert screen("X", {"funding_types": ["equity investment"]}, p, TODAY).status == "Not a lead"


def test_early_stage_only_is_conditional(full_profile):
    assert screen("X", {"stages_allowed": ["idea or pre-seed"]}, full_profile, TODAY).status == "Conditional lead"
    assert screen("X", {"stages_allowed": ["seed", "growth"]}, full_profile, TODAY).status == "Lead"


def test_founder_identity_is_conditional(full_profile):
    v = screen("X", {"founder_identity_required": ["veteran"]}, full_profile, TODAY)
    assert v.status == "Conditional lead" and "veteran" in v.conditions[0]


# --- The Ardook Data profile ---------------------------------------------------------------


def test_ardook_data_profile_loads_and_rates_every_category():
    from ardook_grants.checklist import THEME_CATEGORIES, build_groups
    from ardook_grants.profile import ARDOOK_DATA_PROFILE

    p = load_profile(ARDOOK_DATA_PROFILE)
    build_groups(p)  # raises if the categories don't match
    rated = [c for r in ("good", "partial", "bad") for c in p["theme_categories"][r]]
    assert sorted(rated) == sorted(THEME_CATEGORIES) and p["builds_ai"] is True


def test_both_profiles_ask_the_same_theme_question():
    from ardook_grants.checklist import build_groups
    from ardook_grants.profile import ARDOOK_DATA_PROFILE

    assert build_groups(load_profile())["theme"].version == build_groups(load_profile(ARDOOK_DATA_PROFILE))["theme"].version


@pytest.fixture
def data_profile():
    from ardook_grants.profile import ARDOOK_DATA_PROFILE

    return load_profile(ARDOOK_DATA_PROFILE)


def test_ardook_data_known_legal_form(data_profile):
    assert screen("X", {"applicant_types": ["nonprofit"]}, data_profile, TODAY).status == "Not a lead"
    assert screen("X", {"applicant_types": ["for-profit company"]}, data_profile, TODAY).status == "Lead"


def test_ardook_data_us_and_ai_are_fine_but_armenia_only_is_not(data_profile):
    assert screen("X", {"applicant_countries": ["United States"]}, data_profile, TODAY).status == "Lead"
    assert screen("X", {"program_focus_is_ai_or_deep_tech": "yes"}, data_profile, TODAY).status == "Lead"
    assert screen("X", {"applicant_countries": ["Armenia"]}, data_profile, TODAY).status == "Not a lead"


def test_new_entity_meets_age_limits(data_profile):
    assert screen("X", {"max_company_age_years": 2}, data_profile, TODAY).status == "Lead"


@pytest.mark.parametrize("place", ["U.S.", "US", "USA", "the United States", "U.S.-based companies"])
def test_us_abbreviations_match_a_us_profile(place, data_profile):
    assert screen("X", {"applicant_countries": [place]}, data_profile, TODAY).status == "Lead"


def test_us_abbreviation_still_excluded_for_ardook(profile):
    assert screen("X", {"applicant_countries": ["U.S."]}, profile, TODAY).status == "Not a lead"


# --- Regions, minimum years and minimum revenue (the "limits" questions) -------------------


@pytest.fixture
def ardook():
    return load_profile()  # the real profile, with operating_places


@pytest.mark.parametrize(
    "regions, status",
    [
        (["Aragatsotn Province"], "Not a lead"),            # Armenian province Ardook doesn't work in
        (["Tavush", "Lori", "Gegharkunik"], "Not a lead"),
        (["Yerevan"], "Lead"),
        (["Kotayk Province"], "Lead"),                       # Abovyan's province
        (["Shirak"], "Conditional lead"),                    # Gyumri, planned
        (["Armenia"], "Lead"),                               # just the country: no regional limit
        (["worldwide"], "Lead"),
    ],
)
def test_regions_within_armenia(regions, status, ardook):
    ex = {"applicant_countries": ["Armenia"], "applicant_regions": regions}
    assert screen("X", ex, ardook, TODAY).status == status


def test_region_rule_skipped_without_geography(ardook):
    ex = {"applicant_countries": ["Armenia"], "applicant_regions": ["Aragatsotn"]}
    assert screen("X", ex, ardook, TODAY, use_geography=False).status == "Lead"


def test_us_state_limits_for_ardook_data(data_profile):
    assert screen("X", {"applicant_countries": ["United States"], "applicant_regions": ["California"]},
                  data_profile, TODAY).status == "Lead"
    assert screen("X", {"applicant_countries": ["United States"], "applicant_regions": ["Texas", "Nebraska"]},
                  data_profile, TODAY).status == "Not a lead"


def test_minimum_years(ardook, data_profile):
    # Ardook: established 2020, launched 2022 -> 6 or 4 years in 2026
    assert screen("X", {"min_years_in_operation": 1}, ardook, TODAY).status == "Lead"
    assert screen("X", {"min_years_in_operation": 5}, ardook, TODAY).status == "Conditional lead"
    assert screen("X", {"min_years_in_operation": 10}, ardook, TODAY).status == "Not a lead"
    assert screen("X", {"min_years_in_operation": 1}, data_profile, TODAY).status == "Conditional lead"


def test_minimum_revenue_is_conditional_when_revenue_unknown(ardook):
    v = screen("X", {"min_annual_revenue_usd": 75000}, ardook, TODAY)
    assert v.status == "Conditional lead" and "at least $75,000" in v.conditions[0]


@pytest.mark.parametrize(
    "regions, status",
    [
        (["Aragatsotn Province (Ashtarak area), Armenia"], "Not a lead"),  # country named inside the region
        (["U.S.", "Europe", "Asia"], "Lead"),                              # countries and continents aren't regions
    ],
)
def test_region_answers_that_mention_countries(regions, status, ardook, data_profile):
    p = ardook if "Aragatsotn" in regions[0] else data_profile
    countries = ["Armenia"] if p is ardook else ["United States"]
    assert screen("X", {"applicant_countries": countries, "applicant_regions": regions}, p, TODAY).status == status
