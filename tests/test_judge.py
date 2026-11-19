"""Tests for the retrieval-augmented judge, the synthetic Ardook Data generator and the Week 9 rule fixes.

None of these load a model or call a server.
"""

import random
import sys
from pathlib import Path

import pytest

from ardook_grants.checklist import apply_word_guards
from ardook_grants.judge import Example, judge_prompt, vote
from ardook_grants.profile import ARDOOK_DATA_PROFILE, load_profile
from ardook_grants.rules import classify_location, screen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import make_synthetic_ardook_data as synth  # noqa: E402


def test_vote_majority_and_share():
    answers = [{"fit": "High"}, {"fit": "High", "reason": "r"}, {"fit": "Low"}]
    fit, share, _ = vote(answers)
    assert fit == "High" and share == pytest.approx(2 / 3)


def test_vote_tie_goes_to_moderate_then_lower():
    assert vote([{"fit": "High"}, {"fit": "Moderate"}])[0] == "Moderate"
    assert vote([{"fit": "High"}, {"fit": "Low"}])[0] == "Low"


def test_vote_ignores_invalid_answers():
    assert vote([{"_error": "bad json"}, {"fit": "nonsense"}]) == (None, 0.0, "")
    assert vote([{"_error": "x"}, {"fit": "Strong", "reason": "ok"}])[0] == "High"  # sheet words map to tiers


def test_prompt_contains_examples_and_grant():
    ex = [Example("Program: A", "High", "AI startups", "sponsor")]
    p = judge_prompt("Program: B", "profile text", ex)
    assert "Program: A" in p and "Rating: High (AI startups)" in p and p.rstrip().endswith("Program: B")


def test_synthetic_gold_labels_follow_the_facts():
    rng = random.Random(0)
    for _ in range(200):
        facts = synth.sample_facts(rng)
        label, decisive = synth.gold(facts)
        if facts["bad_slot"]:
            assert label == "Low" and decisive.startswith(facts["bad_slot"])
        else:
            assert label == {"core": "High", "adjacent": "Moderate", "off": "Low"}[facts["theme_kind"]]


def test_synthetic_validation_needs_the_deciding_words():
    facts = {"theme": ("AI startups", r"\bAI\b"), "applicant": ("nonprofits only", r"nonprofit"), "geo": ("x", None),
             "stage": ("x", None), "bad_slot": "applicant"}
    good = "Program: P\nType: Grant\nAmount / Benefit: $1\nCore Requirements: AI projects by nonprofit groups only."
    assert synth.valid(good, facts)
    assert not synth.valid(good.replace("nonprofit", "community"), facts)  # dropped the deciding fact
    assert not synth.valid(good + " Ardook", facts)  # names the test subject


def test_nonprofit_answer_dropped_without_nonprofit_wording():
    facts = {"applicant_types": ["nonprofit", "for-profit company"]}
    problems = apply_word_guards(facts, "Women-led startups with an MVP; seed stage")
    assert facts["applicant_types"] == ["for-profit company"] and problems


def test_nonprofit_answer_kept_when_stated():
    facts = {"applicant_types": ["nonprofit"]}
    apply_word_guards(facts, "Open to registered 501(c)(3) organizations")
    assert facts["applicant_types"] == ["nonprofit"]


@pytest.fixture
def data_profile():
    return load_profile(ARDOOK_DATA_PROFILE)


@pytest.mark.parametrize("place", ["California", "SF", "San Francisco Bay Area"])
def test_operating_places_count_as_eligible_locations(data_profile, place):
    assert classify_location(place, data_profile) == "eligible"
    assert screen("X", {"applicant_countries": [place]}, data_profile).status != "Not a lead"


def test_other_states_still_excluded(data_profile):
    assert screen("X", {"applicant_countries": ["Kenya"]}, data_profile).status == "Not a lead"
