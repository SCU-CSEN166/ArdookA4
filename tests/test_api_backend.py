"""The hosted-model option, tested against a fake HTTP layer (no key, no network)."""

import io
import json
import urllib.error

import pytest

from ardook_grants import backends
from ardook_grants.backends import APIBackend, APIError, read_secret
from ardook_grants.checklist import build_groups, parse_json_answer
from ardook_grants.profile import load_profile


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("https://x", code, "err", headers, io.BytesIO(b"{}"))


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "akml-test")
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    return APIBackend("some/model", "https://example.test/v1", api_key_env="TEST_KEY")


def test_success_sends_bearer_key_and_parses_json(api, monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.data)
        content = '```json\n{"program_focus_is_ai_or_deep_tech": "no"}\n```'
        return FakeResponse(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    monkeypatch.setattr(backends.urllib.request, "urlopen", fake_urlopen)
    answer = api.ask("Program: X", build_groups(load_profile())["ai"])
    assert answer == {"program_focus_is_ai_or_deep_tech": "no"}
    assert seen["auth"] == "Bearer akml-test"
    assert seen["body"]["model"] == "some/model" and seen["body"]["temperature"] == 0


def test_rate_limit_is_retried(api, monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise http_error(429, retry_after="1")
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())

    monkeypatch.setattr(backends.urllib.request, "urlopen", fake_urlopen)
    assert api.generate("s", "u") == "{}" and len(calls) == 3


@pytest.mark.parametrize("code, message", [(402, "credits"), (401, "rejected")])
def test_billing_and_key_errors_stop_the_run(api, monkeypatch, code, message):
    def fake_urlopen(request, timeout):
        raise http_error(code)

    monkeypatch.setattr(backends.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(APIError, match=message):
        api.generate("s", "u")


def test_missing_key_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    monkeypatch.setattr(backends, "REPO_ROOT", tmp_path)
    with pytest.raises(APIError, match="NO_SUCH_KEY"):
        APIBackend(api_key_env="NO_SUCH_KEY").generate("s", "u")


def test_key_can_come_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setattr(backends, "REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text("OTHER=1\nMY_KEY='akml-abc'\n")
    assert read_secret("MY_KEY") == "akml-abc"


def test_reasoning_text_is_ignored():
    assert parse_json_answer('<think>maybe {"x": 1}?</think>{"fit": "High"}') == {"fit": "High"}


def test_no_thinking_sends_vllm_template_flag(api, monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())

    monkeypatch.setattr(backends.urllib.request, "urlopen", fake_urlopen)
    api.extra = APIBackend(no_thinking=True).extra
    api.generate("s", "u")
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_parallel_prefetch_fills_the_cache(tmp_path):
    from ardook_grants.backends import AnswerCache
    from ardook_grants.data import load_json
    from ardook_grants.pipeline import GROUP_ORDER, prefetch, run
    from ardook_grants.profile import REPO_ROOT

    class Echo:
        calls = 0

        def ask(self, text, group):
            Echo.calls += 1
            return {}

    profile = load_profile()
    groups = build_groups(profile)
    grants = load_json(REPO_ROOT / "examples" / "synthetic_grants.json")
    cache = AnswerCache(tmp_path / "c.json")
    assert prefetch(grants, groups, Echo(), cache, workers=4) == len(grants) * len(GROUP_ORDER)
    assert prefetch(grants, groups, Echo(), cache, workers=4) == 0  # all cached now
    results = run(grants, groups, profile, None, cache, rules_only=True, verbose=False)
    assert len(results) == 2 and Echo.calls == len(grants) * len(GROUP_ORDER)
