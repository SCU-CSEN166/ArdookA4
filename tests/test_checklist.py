import pytest

from ardook_grants.checklist import build_groups, clean, parse_json_answer, quote_found, validate
from ardook_grants.profile import load_profile

TEXT = (
    "Program: Hearthside Grant\nType: Loan\nAmount / Benefit: $10,000\n"
    "Core Requirements: Registered nonprofits in Armenia; projects creating home-based jobs for mothers"
)


@pytest.fixture(scope="module")
def groups():
    return build_groups(load_profile())


def test_groups_build_from_the_real_profile(groups):
    assert set(groups) == {"applicants", "ai", "requirements", "theme", "limits"}
    assert "disability inclusion" in groups["theme"].template["focus_areas"][0]


def test_version_changes_when_the_question_changes(groups):
    g = groups["ai"]
    changed = type(g)(g.name, {"program_focus_is_ai_or_deep_tech": ["yes", "no", "maybe"]}, g.examples)
    assert changed.version != g.version


def test_validate_keeps_allowed_values_and_numbers(groups):
    facts, problems = validate(groups["applicants"], {
        "applicant_types": ["Nonprofit", "charity"],
        "women_led_required": "Yes",
        "min_women_ownership_percent": "51%",
        "max_annual_revenue_usd": "$2M",
        "max_company_age_years": "5",
        "applicant_countries": "Armenia",
    })
    assert facts["applicant_types"] == ["nonprofit"]
    assert facts["women_led_required"] == "yes"
    assert facts["min_women_ownership_percent"] == 51
    assert facts["max_annual_revenue_usd"] == 2_000_000
    assert facts["max_company_age_years"] == 5
    assert facts["applicant_countries"] == ["Armenia"]
    assert facts["impact_countries"] == []
    assert any("charity" in p for p in problems)


def test_validate_drops_bad_enum_and_number(groups):
    facts, problems = validate(groups["applicants"], {"women_led_required": "maybe", "max_company_age_years": "young"})
    assert facts["women_led_required"] is None and facts["max_company_age_years"] is None
    assert len(problems) == 2


@pytest.mark.parametrize(
    "quote, found",
    [
        ("Registered nonprofits in Armenia", True),
        ("registered  NONPROFITS in armenia", True),  # spacing and case don't matter
        ("Registered nonprofits in Armenia ... home-based jobs for mothers", True),  # parts joined by ...
        ("projects creating home based jobs for mothers", True),  # one character off
        ("Registered businesses in Georgia", False),
        ("...", True),  # nothing to check
    ],
)
def test_quote_found(quote, found):
    assert quote_found(quote, TEXT) is found


def test_made_up_quote_voids_the_answers_it_supports(groups):
    raw = {"applicant_countries": ["United States"], "applicant_types": ["for-profit company"],
           "applicant_evidence": ["Open to U.S. companies only"], "award": "$10,000"}
    facts, report = clean(groups["applicants"], raw, TEXT)
    assert facts["applicant_countries"] == [] and facts["applicant_types"] == []
    assert facts["award"] == "$10,000"  # not backed by the evidence field, so kept
    assert set(report["unverified"]) == {"applicant_countries", "applicant_types"}


def test_real_quote_keeps_the_answers(groups):
    raw = {"applicant_countries": ["Armenia"], "applicant_evidence": ["Registered nonprofits in Armenia"]}
    facts, report = clean(groups["applicants"], raw, TEXT)
    assert facts["applicant_countries"] == ["Armenia"] and report["unverified"] == []


def test_guessed_stage_is_dropped_when_the_text_never_mentions_one(groups):
    raw = {"funding_types": ["loan"], "stages_allowed": ["seed"], "founder_identity_required": ["veteran"]}
    facts, report = clean(groups["requirements"], raw, TEXT)
    assert facts["funding_types"] == ["loan"]  # "Loan" is in the text
    assert facts["stages_allowed"] == [] and facts["founder_identity_required"] == []
    assert len(report["problems"]) == 2


def test_model_error_is_reported_not_raised(groups):
    facts, report = clean(groups["theme"], {"_error": "model output was not valid JSON"}, TEXT)
    assert facts == {} and report["error"]


@pytest.mark.parametrize(
    "raw, ok",
    [('{"a": 1}', True), ('```json\n{"a": 1}\n```', True), ('Sure! {"a": 1} done', True), ("not json", False)],
)
def test_parse_json_answer(raw, ok):
    assert ("_error" not in parse_json_answer(raw)) is ok


def test_nested_multi_select_is_flattened(groups):
    facts, problems = validate(groups["applicants"], {"applicant_types": [["for-profit company", "nonprofit"]]})
    assert facts["applicant_types"] == ["for-profit company", "nonprofit"] and problems == []
