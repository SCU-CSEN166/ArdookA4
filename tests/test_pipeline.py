"""End-to-end run with a fake model, so CI tests the whole pipeline without downloading anything."""

import csv
import json

import pytest

from ardook_grants.backends import AnswerCache
from ardook_grants.checklist import build_groups
from ardook_grants.data import load_json
from ardook_grants.pipeline import parse_deadline, process, run, shortlist, write_outputs
from ardook_grants.profile import REPO_ROOT, load_profile

EXAMPLES = REPO_ROOT / "examples" / "synthetic_grants.json"

# What a model would answer for the two example grants (quotes copied from their text).
ANSWERS = {
    "Caucasus Inclusive Livelihoods Fund": {
        "applicants": {"applicant_countries": ["Armenia", "Georgia", "Azerbaijan"], "applicant_types": ["nonprofit"],
                       "applicant_evidence": ["Registered non-governmental organizations in Armenia, Georgia or Azerbaijan"]},
        "ai": {"program_focus_is_ai_or_deep_tech": "no"},
        "requirements": {"funding_types": ["grant"], "requirements_evidence": ["Type: Grant"]},
        "theme": {"focus_areas": ["jobs or income for mothers and caregivers", "disability inclusion"],
                  "focus_evidence": ["home-based employment for caregivers of people with disabilities"]},
    },
    "Main Street Women Founders Award": {
        "applicants": {"applicant_countries": ["United States"], "applicant_types": ["for-profit company"],
                       "min_women_ownership_percent": 51, "max_company_age_years": 5,
                       "applicant_evidence": ["for-profit small business based in the United States"]},
        "ai": {"program_focus_is_ai_or_deep_tech": "no"},
        "requirements": {"funding_types": ["grant"]},
        "theme": {"focus_areas": ["women entrepreneurs in any industry"], "focus_evidence": ["Woman-owned"]},
    },
}


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def ask(self, text, group):
        self.calls += 1
        program = text.split("\n")[0].removeprefix("Program: ")
        return ANSWERS[program].get(group.name, {})


@pytest.fixture(scope="module")
def setup():
    profile = load_profile()
    return load_json(EXAMPLES), build_groups(profile), profile


def test_end_to_end_with_fake_model(setup, tmp_path):
    grants, groups, profile = setup
    backend, cache = FakeBackend(), AnswerCache(tmp_path / "cache.json")
    results = run(grants, groups, profile, backend, cache, verbose=False)
    by_name = {r.program: r for r in results}

    care = by_name["Caucasus Inclusive Livelihoods Fund"]
    assert care.eligibility == "Conditional lead" and "registered nonprofit" in care.check
    assert care.theme == "good" and care.entity == "partial" and care.fit == "Moderate"
    assert all(e.startswith("✓") for e in care.evidence.split(" | "))

    main_street = by_name["Main Street Women Founders Award"]
    assert main_street.eligibility == "Not a lead" and main_street.location == "excluded"
    assert main_street.fit == "Moderate"  # location doesn't lower Fit

    # Expected eligibility written into the example file
    for g, r in zip(grants, results):
        assert g["sheet_fit"].startswith(r.eligibility)

    write_outputs(results, tmp_path / "out")
    rows = list(csv.DictReader((tmp_path / "out" / "results.csv").open(encoding="utf-8")))
    assert len(rows) == 2 and "trace" not in rows[0]
    assert [r["program"] for r in csv.DictReader((tmp_path / "out" / "shortlist.csv").open())] == [care.program]
    assert len((tmp_path / "out" / "trace.jsonl").read_text().splitlines()) == 2


def test_cached_answers_need_no_model(setup, tmp_path):
    grants, groups, profile = setup
    cache = AnswerCache(tmp_path / "cache.json")
    run(grants, groups, profile, FakeBackend(), cache, verbose=False)
    cache.save()

    again = FakeBackend()
    results = run(grants, groups, profile, again, AnswerCache(tmp_path / "cache.json"), rules_only=True, verbose=False)
    assert again.calls == 0 and len(results) == 2

    empty = run(grants, groups, profile, again, AnswerCache(tmp_path / "none.json"), rules_only=True, verbose=False)
    assert empty == []


def test_made_up_quote_makes_the_grant_conditional(setup):
    grants, groups, profile = setup
    raw = json.loads(json.dumps(ANSWERS["Caucasus Inclusive Livelihoods Fund"]))
    raw["applicants"]["applicant_types"] = ["any organization"]
    raw["applicants"]["applicant_evidence"] = ["Open to any organization worldwide"]
    r = process(grants[0], raw, groups, profile)
    assert r.eligibility == "Conditional lead" and r.needs_review and "check by hand" in r.check


def test_failed_extraction_is_flagged_not_excluded(setup):
    grants, groups, profile = setup
    raw = {**ANSWERS["Caucasus Inclusive Livelihoods Fund"], "applicants": {"_error": "model output was not valid JSON"}}
    r = process(grants[0], raw, groups, profile)
    assert r.eligibility == "Conditional lead" and r.needs_review


def test_shortlist_order(setup, tmp_path):
    grants, groups, profile = setup
    results = run(grants, groups, profile, FakeBackend(), AnswerCache(tmp_path / "cache.json"), verbose=False)
    assert [r.program for r in shortlist(results)] == ["Caucasus Inclusive Livelihoods Fund"]


def test_parse_deadline():
    assert parse_deadline("2026-09-16").isoformat() == "2026-09-16"
    assert parse_deadline("11-12-26").isoformat() == "2026-11-12"
    assert parse_deadline("1/15/2027").isoformat() == "2027-01-15"
    assert parse_deadline("Rolling") is None and parse_deadline("13-45-26") is None


def test_limits_answers_replace_confused_applicant_ones(setup):
    grants, groups, profile = setup
    raw = json.loads(json.dumps(ANSWERS["Main Street Women Founders Award"]))
    raw["applicants"]["max_company_age_years"] = 18  # "under 18 months" misread as years
    raw["limits"] = {"max_years_in_operation": 1.5, "limits_evidence": ["under 5 years old"]}
    r = process(grants[1], raw, groups, profile, use_geography=False)
    assert "under 1.5 years" in r.eligibility_why or "1.5" in r.check or r.eligibility == "Not a lead"
    assert r.trace["facts"]["max_company_age_years"] == 1.5


def test_runs_without_limits_answers_still_work(setup):
    grants, groups, profile = setup
    raw = {k: v for k, v in ANSWERS["Caucasus Inclusive Livelihoods Fund"].items()}
    r = process(grants[0], raw, groups, profile)  # no "limits" key, as in caches from before it existed
    assert r.eligibility == "Conditional lead" and not r.needs_review
