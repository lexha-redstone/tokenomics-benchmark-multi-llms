# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for the upstream grading-machinery cache.

Why this exists
---------------
`run_script.sh` and `parser.py` are fetched per instance from upstream and
cached under `swebench_pro/run_scripts/<instance_id>/`. They *are* the grading:
the first runs the repository's suite, the second decides which test names
passed. If either is wrong, every arm on that row scores zero and the results
file cannot say why.

The cache used to write first and check afterwards. A body that was not the
file — a 404 page, a truncated read — was persisted, then rejected; every later
run found the file present, skipped the fetch, re-read the same bad bytes and
raised again. One transient fault pinned an instance at "unavailable"
permanently. `parser.py` had no check at all, so its version of the same fault
was quieter and worse: no `output.json` is written, `sbp_resolution` reads
`{}`, and the row grades as `reported: 0` — indistinguishable in the results
file from a candidate patch that broke the suite.

The live checkout still carries the evidence: one instance directory under
`swebench_pro/run_scripts/` is empty, left behind by a fetch that failed after
`makedirs`.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import datasets as ds                                        # noqa: E402


GOOD_RUN_SCRIPT = "#!/bin/bash\nrun_all_tests() { pytest \"$@\"; }\nrun_all_tests\n"
GOOD_PARSER = "import json\ndef parse(a, b, out):\n    json.dump({}, open(out, 'w'))\n"
NOT_FOUND = "<!DOCTYPE html><title>404: Not Found</title>"


@pytest.fixture
def fetches(monkeypatch):
    """Record every URL fetched and serve canned bodies."""
    calls = []
    bodies = {}

    def fake_fetch(url, timeout=180, max_retries=6):
        calls.append(url)
        if url not in bodies:
            raise RuntimeError(f"HTTP 404 for {url}")
        return bodies[url].encode("utf-8")

    monkeypatch.setattr(ds, "_fetch", fake_fetch)
    return calls, bodies


# ==============================================================================
# --- VALIDATE BEFORE PERSISTING ---
# ==============================================================================

def test_a_body_that_fails_validation_is_never_written(tmp_path, fetches):
    calls, bodies = fetches
    url = "https://example/run_script.sh"
    bodies[url] = NOT_FOUND
    path = tmp_path / "inst" / "run_script.sh"

    with pytest.raises(RuntimeError, match="does not look like"):
        ds._sbp_cached(url, str(path), validate=ds._looks_like_run_script)
    assert not path.exists(), "a rejected body must not become tomorrow's cache"


def test_a_valid_body_is_cached_and_returned(tmp_path, fetches):
    calls, bodies = fetches
    url = "https://example/run_script.sh"
    bodies[url] = GOOD_RUN_SCRIPT
    path = tmp_path / "inst" / "run_script.sh"

    assert ds._sbp_cached(url, str(path),
                          validate=ds._looks_like_run_script) == GOOD_RUN_SCRIPT
    assert path.read_text(encoding="utf-8") == GOOD_RUN_SCRIPT


def test_the_cache_is_used_on_the_second_call(tmp_path, fetches):
    calls, bodies = fetches
    url = "https://example/run_script.sh"
    bodies[url] = GOOD_RUN_SCRIPT
    path = tmp_path / "inst" / "run_script.sh"
    ds._sbp_cached(url, str(path), validate=ds._looks_like_run_script)
    ds._sbp_cached(url, str(path), validate=ds._looks_like_run_script)
    assert len(calls) == 1


def test_no_temporary_file_is_left_behind(tmp_path, fetches):
    """The write is atomic, so a crash mid-write cannot leave a half file that
    the next run reads as content."""
    calls, bodies = fetches
    url = "https://example/run_script.sh"
    bodies[url] = GOOD_RUN_SCRIPT
    path = tmp_path / "inst" / "run_script.sh"
    ds._sbp_cached(url, str(path), validate=ds._looks_like_run_script)
    assert os.listdir(path.parent) == ["run_script.sh"]


# ==============================================================================
# --- A POISONED CACHE HEALS INSTEAD OF RAISING FOREVER ---
# ==============================================================================

def test_a_cache_entry_that_fails_its_validator_is_refetched(tmp_path, fetches):
    """The regression: before this, a bad file on disk was re-read and
    re-rejected on every run, so the instance never recovered without somebody
    deleting the directory by hand."""
    calls, bodies = fetches
    url = "https://example/run_script.sh"
    path = tmp_path / "inst" / "run_script.sh"
    path.parent.mkdir(parents=True)
    path.write_text(NOT_FOUND, encoding="utf-8")

    bodies[url] = GOOD_RUN_SCRIPT
    assert ds._sbp_cached(url, str(path),
                          validate=ds._looks_like_run_script) == GOOD_RUN_SCRIPT
    assert calls == [url]
    assert path.read_text(encoding="utf-8") == GOOD_RUN_SCRIPT


def test_a_poisoned_cache_that_cannot_be_refetched_raises_rather_than_returning_it(
        tmp_path, fetches):
    calls, bodies = fetches
    url = "https://example/run_script.sh"
    path = tmp_path / "inst" / "run_script.sh"
    path.parent.mkdir(parents=True)
    path.write_text(NOT_FOUND, encoding="utf-8")
    with pytest.raises(RuntimeError):
        ds._sbp_cached(url, str(path), validate=ds._looks_like_run_script)


def test_no_validator_means_the_old_behaviour_is_unchanged(tmp_path, fetches):
    """The dockerfile fetches pass no validator and must keep working."""
    calls, bodies = fetches
    url = "https://example/Dockerfile"
    bodies[url] = "FROM scratch\nENV A=1\n"
    path = tmp_path / "inst" / "base.Dockerfile"
    assert "FROM scratch" in ds._sbp_cached(url, str(path))


# ==============================================================================
# --- THE VALIDATORS THEMSELVES ---
# ==============================================================================

def test_the_run_script_validator_requires_upstreams_entry_point():
    assert ds._looks_like_run_script(GOOD_RUN_SCRIPT) is True
    assert ds._looks_like_run_script(NOT_FOUND) is False
    assert ds._looks_like_run_script("") is False


def test_the_parser_validator_exists_at_all():
    """It did not, and that was the quieter half of the bug: a bad `parser.py`
    produces no verdict file, and `reported: 0` reads as a broken suite."""
    assert ds._looks_like_parser(GOOD_PARSER) is True
    assert ds._looks_like_parser(NOT_FOUND) is False
    assert ds._looks_like_parser("") is False


# ==============================================================================
# --- THE CALLER WIRES BOTH VALIDATORS UP ---
# ==============================================================================

def test_ensure_scripts_validates_the_parser_as_well_as_the_run_script(
        tmp_path, monkeypatch, fetches):
    calls, bodies = fetches
    monkeypatch.setattr(ds, "swebench_pro_scripts_dir",
                        lambda iid: str(tmp_path / iid))
    monkeypatch.setattr(ds, "_sbp_env_exports", lambda iid, refresh=False: "")
    base = f"{ds.SWEBENCH_PRO_SCRIPTS_RAW}/run_scripts/inst"
    bodies[f"{base}/run_script.sh"] = GOOD_RUN_SCRIPT
    bodies[f"{base}/parser.py"] = NOT_FOUND

    with pytest.raises(RuntimeError, match="does not look like"):
        ds.ensure_swebench_pro_scripts("inst")
    assert not (tmp_path / "inst" / "parser.py").exists()


def test_ensure_scripts_returns_both_files_when_upstream_is_healthy(
        tmp_path, monkeypatch, fetches):
    calls, bodies = fetches
    monkeypatch.setattr(ds, "swebench_pro_scripts_dir",
                        lambda iid: str(tmp_path / iid))
    monkeypatch.setattr(ds, "_sbp_env_exports", lambda iid, refresh=False: "export A=1")
    base = f"{ds.SWEBENCH_PRO_SCRIPTS_RAW}/run_scripts/inst"
    bodies[f"{base}/run_script.sh"] = GOOD_RUN_SCRIPT
    bodies[f"{base}/parser.py"] = GOOD_PARSER

    got = ds.ensure_swebench_pro_scripts("inst")
    assert got["run_script"] == GOOD_RUN_SCRIPT
    assert got["parser"] == GOOD_PARSER
    assert got["env_exports"] == "export A=1"
