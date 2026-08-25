# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Contract tests for the dispatch failure policy (`src/client.py`).

The regression these exist for: every unrecoverable API failure used to be
answered with `_fallback_dispatch`, a simulator that priced its invented token
counts with the real rate card and returned them in the ordinary usage dict.
A 504 or an expired credential therefore became an ordinary-looking benchmark
datapoint, indistinguishable from a live one after the fact.

Pinned here:

  * a failure raises rather than fabricating;
  * simulation happens only when explicitly requested, and is marked when it
    does;
  * transient and auth failures are classified as retryable, permanent ones
    are not;
  * an auth failure forces a token refresh before retrying;
  * the retry budget is actually spent.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.client as client  # noqa: E402
from src.client import DispatchError, classify_error  # noqa: E402


class _HttpLike(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


@pytest.fixture(autouse=True)
def _no_simulation(monkeypatch):
    """Default posture for every test unless one opts in."""
    monkeypatch.delenv("ALLOW_SIMULATION", raising=False)


# ------------------------------------------------------------ classification

@pytest.mark.parametrize("exc,expected", [
    (_HttpLike("504 Gateway Timeout", 504), "transient"),
    (_HttpLike("502 Bad Gateway", 502), "transient"),
    (_HttpLike("503 Service Unavailable"), "transient"),
    (_HttpLike("429 Too Many Requests", 429), "transient"),
    (TimeoutError("timed out"), "transient"),
    (_HttpLike("connection reset by peer"), "transient"),
    (_HttpLike("401 Unauthorized", 401), "auth"),
    (_HttpLike("403 Permission denied", 403), "auth"),
    (_HttpLike("Reauthentication is needed; invalid_grant"), "auth"),
    (_HttpLike("Your default credentials have expired"), "auth"),
    (ValueError("model does not exist"), "permanent"),
    (KeyError("bad field"), "permanent"),
])
def test_error_classification(exc, expected):
    assert classify_error(exc) == expected


# --------------------------------------------------------------- refuse first

def test_failure_raises_instead_of_fabricating():
    """The core rule. A failed call must not become a datapoint."""
    with pytest.raises(DispatchError) as ei:
        client._give_up("gemini-3.7-flash", _HttpLike("504", 504), "transient", 5,
                        "prompt", 100, None, None)
    assert ei.value.kind == "transient"
    assert ei.value.attempts == 5
    assert "discarded and retried" in str(ei.value)


def test_auth_failure_names_the_fix():
    with pytest.raises(DispatchError) as ei:
        client._give_up("claude-opus-5", _HttpLike("401", 401), "auth", 3,
                        "prompt", 100, None, None)
    assert "gcloud auth application-default login" in str(ei.value)


def test_simulation_is_opt_in_and_marked(monkeypatch):
    monkeypatch.setenv("ALLOW_SIMULATION", "1")
    text, usage, _ = client._give_up(
        "gemini-3.7-flash", _HttpLike("504", 504), "transient", 5,
        "Problem:\n```python\ndef task_func(x): pass\n```", 100, None, None)
    assert text
    assert usage["simulated"] is True, "simulated output must be distinguishable"


def test_live_usage_is_never_marked_simulated():
    """The marker must mean something: real calls do not carry it."""
    _, usage, _ = client._fallback_dispatch("gemini-3.7-flash", "prompt")
    assert "simulated" not in usage, (
        "_fallback_dispatch itself stays unmarked; the marker is stamped by "
        "_give_up, which is the only sanctioned door to it")


# ------------------------------------------------------------------- retrying

def test_transient_failures_are_retried_then_raised(monkeypatch):
    """The budget is spent before giving up, and the count is reported."""
    calls = {"n": 0}

    def always_504(req, timeout=None, context=None):
        calls["n"] += 1
        raise _HttpLike("504 Gateway Timeout", 504)

    monkeypatch.setattr(client, "_vertex_access_token", lambda *a, **k: "tok")
    monkeypatch.setattr(client.urllib.request, "urlopen", always_504)
    monkeypatch.setattr(client, "_sleep_for", lambda attempt: 0.0)

    with pytest.raises(DispatchError) as ei:
        client.claude_api_call("claude-opus-5", "prompt")
    assert calls["n"] == client.MAX_ATTEMPTS
    assert ei.value.attempts == client.MAX_ATTEMPTS


def test_auth_failure_forces_a_token_refresh_before_retrying(monkeypatch):
    """A cached token may be exactly what the server rejected."""
    refreshes = {"forced": 0}

    def token(force_refresh=False):
        if force_refresh:
            refreshes["forced"] += 1
        return "tok"

    def always_401(req, timeout=None, context=None):
        raise _HttpLike("401 Unauthorized", 401)

    monkeypatch.setattr(client, "_vertex_access_token", token)
    monkeypatch.setattr(client.urllib.request, "urlopen", always_401)
    monkeypatch.setattr(client, "_sleep_for", lambda attempt: 0.0)

    with pytest.raises(DispatchError):
        client.claude_api_call("claude-opus-5", "prompt")
    assert refreshes["forced"] >= 1


def test_permanent_failure_is_not_retried(monkeypatch):
    """Retrying a malformed request just burns time."""
    calls = {"n": 0}

    def bad_request(req, timeout=None, context=None):
        calls["n"] += 1
        raise ValueError("model does not exist")

    monkeypatch.setattr(client, "_vertex_access_token", lambda *a, **k: "tok")
    monkeypatch.setattr(client.urllib.request, "urlopen", bad_request)
    monkeypatch.setattr(client, "_sleep_for", lambda attempt: 0.0)

    with pytest.raises(DispatchError) as ei:
        client.claude_api_call("claude-opus-5", "prompt")
    assert calls["n"] == 1
    assert ei.value.kind == "permanent"


def test_missing_credentials_raise_rather_than_simulate(monkeypatch):
    monkeypatch.setattr(client, "_vertex_access_token", lambda *a, **k: None)
    with pytest.raises(DispatchError) as ei:
        client.claude_api_call("claude-opus-5", "prompt")
    assert ei.value.kind == "auth"


# ------------------------------------------------------- arm-level propagation

def test_arms_propagate_dispatch_errors(monkeypatch):
    """An arm must not swallow the failure and return a partial record — the
    runner relies on the exception to discard and retry the task."""
    import src.architectures as arch
    import src.evaluator as ev

    def boom(model_id, prompt, max_tokens=2048, thinking_level=None, problem=None):
        raise DispatchError("504", model_id=model_id, kind="transient", attempts=5)

    for mod in (client, ev, arch):
        monkeypatch.setattr(mod, "dispatch_model", boom, raising=False)

    problem = {
        "task_id": "t/1", "entry_point": "task_func",
        "complete_prompt": "def task_func(x): return x + 1",
        "test": ("import unittest\nclass TestCases(unittest.TestCase):\n"
                 "    def test_a(self): self.assertEqual(task_func(1), 2)\n"),
    }
    with pytest.raises(DispatchError):
        arch.run_single(dict(problem), model_id="gemini-3.7-flash")


def test_no_silent_fallback_remains_in_the_client():
    """Guard against the pattern coming back: every `_fallback_dispatch` call
    must go through `_give_up`, which enforces the opt-in and the marker."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "src" / "client.py").read_text()
    body = src.split('"""', 2)[-1]
    call_sites = [ln for ln in body.splitlines()
                  if "_fallback_dispatch(" in ln and "def _fallback_dispatch" not in ln]
    for ln in call_sites:
        assert "text, usage, dt = _fallback_dispatch" in ln, (
            f"unsanctioned _fallback_dispatch call site: {ln.strip()}")
    assert "Falling back to simulation" not in src


def test_short_error_str():
    from src.client import _short_error_str

    verbose_429 = (
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Resource exhausted. "
        "Please try again later. Please refer to https://cloud.google.com/vertex-ai for details.', "
        "'status': 'RESOURCE_EXHAUSTED', 'details': [{'detail': '[ORIGINAL ERROR] extensible_stubs...'}]}}"
    )
    s = _short_error_str(verbose_429)
    assert s == "429 RESOURCE_EXHAUSTED: Resource exhausted. Please try again later"
    assert len(s) < 100

    verbose_503 = "503 UNAVAILABLE: Overloaded prefill queue, preempted by higher priority.; details: ..."
    assert _short_error_str(verbose_503) == "503 UNAVAILABLE: Overloaded prefill queue (preempted)"

    assert _short_error_str(_HttpLike("504 Gateway Timeout", 504)) == "504 Gateway Timeout"




# ==============================================================================
# --- THE COST OF GIVING UP TOO EARLY ---
# ==============================================================================
#
# `src/sweep.py` discards a task whose dispatch raised and drops it from the
# arm's denominator. At MAX_ATTEMPTS=1 a single 503 therefore removes a task
# from one arm and leaves it in the others: reports 21 and 23 published arms
# scored over 49, 50, 47, 40, 44 and 37 tasks in one table, side by side, as
# though the denominators matched. The loss is not random either — the tasks
# that time out are the ones with the longest prompts and outputs — so the
# surviving denominator is biased toward the easy rows.

def test_the_retry_budget_is_more_than_one_by_default():
    """One attempt means the first transient blip costs a whole task, and the
    task is silently removed from that arm's denominator only."""
    assert client.MAX_ATTEMPTS >= 3


def test_a_transient_failure_that_later_succeeds_does_not_cost_the_task(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "end_turn"}).encode()

    def flaky(req, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _HttpLike("503 Service Unavailable")
        return _Resp()

    monkeypatch.setattr(client, "_vertex_access_token", lambda *a, **k: "tok")
    monkeypatch.setattr(client.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(client, "_sleep_for", lambda attempt: 0.0)

    text, usage, _ = client.claude_api_call("claude-sonnet-5", "prompt")
    assert text == "ok" and calls["n"] == 2


# ==============================================================================
# --- THE TIMEOUT HAS TO FIT WHAT WAS ASKED FOR ---
# ==============================================================================

def test_the_timeout_scales_with_the_output_budget():
    """A fixed 120s deadline was being applied to `max_tokens=32768` requests.
    At any realistic decode rate it expires while the response is still being
    produced, which arrives as a transient failure and costs the task."""
    small = client.request_timeout(2048)
    large = client.request_timeout(32768)
    assert large > small >= client.DISPATCH_TIMEOUT_FLOOR


def test_the_timeout_never_drops_below_the_floor_or_above_the_cap():
    assert client.request_timeout(0) == client.DISPATCH_TIMEOUT_FLOOR
    assert client.request_timeout(10 ** 9) == client.DISPATCH_TIMEOUT_CAP


def test_a_thirty_two_k_request_gets_a_deadline_that_can_actually_be_met():
    """The value the SWE-bench Pro arms request. 120s was not enough for a
    response the reports themselves show averaging 5k-13k output tokens."""
    assert client.request_timeout(32768) >= 600


def test_the_claude_request_uses_the_scaled_timeout(monkeypatch):
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()

    def capture(req, timeout=None, context=None):
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(client, "_vertex_access_token", lambda *a, **k: "tok")
    monkeypatch.setattr(client.urllib.request, "urlopen", capture)
    client.claude_api_call("claude-sonnet-5", "prompt", max_tokens=32768)
    assert seen["timeout"] == client.request_timeout(32768)


# ==============================================================================
# --- A TRUNCATED ANSWER IS NOT A WRONG ANSWER ---
# ==============================================================================

def _claude_with_stop_reason(monkeypatch, stop_reason):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({
                "content": [{"type": "text", "text": "diff --git"}],
                "usage": {"input_tokens": 10, "output_tokens": 32768},
                "stop_reason": stop_reason}).encode()

    monkeypatch.setattr(client, "_vertex_access_token", lambda *a, **k: "tok")
    monkeypatch.setattr(client.urllib.request, "urlopen",
                        lambda req, timeout=None, context=None: _Resp())
    return client.claude_api_call("claude-sonnet-5", "prompt", max_tokens=32768)


def test_a_response_cut_off_at_the_cap_is_marked_truncated(monkeypatch):
    """Left unrecorded, a diff that stops mid-hunk still carries `---`, `+++`
    and `@@`, passes the birth gate, fails `git apply`, and is filed as the
    harness's most common failure — a budget problem wearing a model's name."""
    _, usage, _ = _claude_with_stop_reason(monkeypatch, "max_tokens")
    assert usage["truncated"] is True
    assert usage["stop_reason"] == "max_tokens"


def test_a_complete_response_is_not_marked_truncated(monkeypatch):
    _, usage, _ = _claude_with_stop_reason(monkeypatch, "end_turn")
    assert usage["truncated"] is False


def test_a_missing_stop_reason_is_not_read_as_truncation(monkeypatch):
    _, usage, _ = _claude_with_stop_reason(monkeypatch, None)
    assert usage["truncated"] is False and usage["stop_reason"] == ""
