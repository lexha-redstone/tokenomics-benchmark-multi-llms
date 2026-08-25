# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for repository grounding on SWE-bench Pro.

Why this exists
---------------
The row hands the model an issue, a requirements block and an interface block,
and asks for a complete unified git diff with real line numbers against a tree
it has never seen. Measured over the published 731-row split:

    reference-patch files named anywhere in those three blocks : 19.8%
    rows where EVERY changed file is named                     :  8.8%
    median reference patch                                     :  9 hunks / 4 files
    patches that only create new files (no context needed)     :  1.8%

So the blind prompt has a localisation ceiling near 9% before a single hunk is
written, and `git apply` needs every context line of every hunk to match. What
these tests pin is the set of things that would make grounding *look* like it
worked while measuring something else:

* the three upstream blocks survive verbatim, so the prompt stays comparable to
  the published leaderboard prompt;
* source is read from `base_commit`, never from the worktree, so one attempt's
  patch cannot leak into the next attempt's prompt;
* the budget is enforced and truncation is announced, so the model is never
  shown an excerpt it believes is a whole file;
* `SBP_GROUNDING_CHARS=0` reproduces the blind prompt exactly, so the change is
  an A/B rather than a one-way door;
* a container that cannot be read degrades to the blind prompt instead of
  failing the task.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import swebench_pro as sbp                                 # noqa: E402
from src.evaluator import SWEBenchProEnv                            # noqa: E402


PROBLEM = {
    "instance_id": "instance_demo__repo-abc123-vnan",
    "repo": "demo/repo",
    "repo_language": "python",
    "base_commit": "b" * 40,
    "problem_statement": "The `loadUserInfo` helper drops the pending flag.",
    "requirements": "`loadUserInfo` must attach `email:pending` to each user.",
    "interface": ("Type: Method\n\nName: db.mget\n\n"
                  "Path: src/database/main.py, src/user/emails.py\n"),
    "fail_to_pass": ["tests/t.py | attaches pending"],
    "pass_to_pass": [],
    "selected_test_files_to_run": ["tests/t.py"],
    "before_repo_set_cmd": "git checkout abc123 -- tests/t.py",
    "dockerhub_tag": "demo.repo-demo__repo-abc123",
    "image_name": "jefzda/sweap-images:demo.repo-demo__repo-abc123",
    "repo_workdir": "/app",
}


class FakeTree:
    """A `SWEBenchProEnv`-shaped object backed by a dict instead of Docker."""

    def __init__(self, files=None, fail=False):
        self.files = dict(files or {})
        self.fail = fail
        self.reads = []
        self.greps = []

    def read_source(self, paths, budget=None, per_file=None, max_files=None):
        if self.fail:
            raise RuntimeError("docker exec: no such container")
        self.reads.append(list(paths))
        blocks, read, skipped = [], [], []
        for p in paths:
            if p in self.files:
                blocks.append(f"--- FILE: {p} ---\n{self.files[p]}\n")
                read.append(p)
            else:
                skipped.append(p)
        return blocks, read, skipped

    def grep_paths(self, terms, limit=40):
        self.greps.append(list(terms))
        return ["src/found_by_grep.py"]


# ==============================================================================
# --- WHERE THE CANDIDATE PATHS COME FROM ---
# ==============================================================================

def test_interface_path_lines_come_first_and_are_split_on_commas():
    """Upstream's own statement of where the new surface lives is the strongest
    signal in the row; anything found by regex over prose is weaker."""
    got = sbp.candidate_paths(PROBLEM)
    assert got[:2] == ["src/database/main.py", "src/user/emails.py"]


def test_graded_test_files_are_candidates():
    """They describe the behaviour being demanded, in the repository's own words."""
    assert "tests/t.py" in sbp.candidate_paths(PROBLEM)


def test_path_shaped_tokens_in_prose_are_picked_up():
    p = dict(PROBLEM, interface="", selected_test_files_to_run=[],
             problem_statement="Broken since we touched lib/handlers/auth.go.")
    assert "lib/handlers/auth.go" in sbp.candidate_paths(p)


def test_candidate_paths_are_deduplicated_and_order_is_stable():
    p = dict(PROBLEM, interface="Path: a/b.py, a/b.py\n",
             selected_test_files_to_run=["a/b.py"],
             problem_statement="see a/b.py")
    assert sbp.candidate_paths(p) == ["a/b.py"]


def test_a_row_that_names_no_path_yields_no_candidates_rather_than_junk():
    p = dict(PROBLEM, interface="", selected_test_files_to_run=[],
             problem_statement="It is broken.", requirements="Fix it.")
    assert sbp.candidate_paths(p) == []


# ==============================================================================
# --- WHAT GETS SEARCHED FOR WHEN THE TEXT NAMES NOTHING ---
# ==============================================================================

def test_search_terms_take_interface_names_and_their_last_segment():
    terms = sbp.search_terms(PROBLEM)
    assert "db.mget" in terms and "mget" in terms


def test_search_terms_take_backticked_identifiers():
    assert "loadUserInfo" in sbp.search_terms(PROBLEM)


def test_search_terms_reject_backticked_prose():
    """A whole sentence in backticks greps to everything or nothing; either way
    it spends a container round trip to learn nothing."""
    p = dict(PROBLEM, interface="", requirements="",
             problem_statement="It fails when `the user has no verified email`.")
    assert sbp.search_terms(p) == []


def test_search_terms_are_bounded():
    p = dict(PROBLEM, interface="", requirements="",
             problem_statement=" ".join(f"`sym{i}`" for i in range(50)))
    assert len(sbp.search_terms(p)) <= 8


# ==============================================================================
# --- COLLECTING THE GROUNDING ---
# ==============================================================================

def test_zero_budget_reproduces_the_blind_prompt_exactly():
    """The A/B leg. `SBP_GROUNDING_CHARS=0` must not merely shrink the block —
    it must make the prompt byte-identical to the one reports 21/23 used, or
    the comparison to them is not a comparison."""
    env = FakeTree({"src/database/main.py": "x = 1\n"})
    text, meta = sbp.collect_grounding(env, PROBLEM, budget=0)
    assert text == "" and meta["enabled"] is False
    assert env.reads == [] and env.greps == []
    assert sbp._solve_prompt(PROBLEM, grounding=text) == sbp._solve_prompt(PROBLEM)


def test_named_files_are_quoted_with_the_base_commit_in_the_header():
    env = FakeTree({"src/database/main.py": "def mget(): ...\n"})
    text, meta = sbp.collect_grounding(env, PROBLEM)
    assert "def mget(): ..." in text
    assert meta["read"] == ["src/database/main.py"]
    assert PROBLEM["base_commit"][:12] in text
    assert "context lines" in text.lower()


def test_a_search_runs_only_when_the_rows_own_text_located_too_little():
    """91% of rows do not name every file they need, so this is the common path
    rather than an exception — but paying for it when the text already worked
    is a container round trip for nothing."""
    plenty = FakeTree({"src/database/main.py": "a\n", "src/user/emails.py": "b\n",
                       "tests/t.py": "c\n"})
    sbp.collect_grounding(plenty, PROBLEM)
    assert plenty.greps == []

    sparse = FakeTree({"src/database/main.py": "a\n"})
    text, meta = sbp.collect_grounding(sparse, PROBLEM)
    assert sparse.greps and "db.mget" in sparse.greps[0]
    assert "src/found_by_grep.py" in meta["skipped"] + meta["read"]
    assert meta["searched"]


def test_files_the_row_named_but_the_tree_does_not_have_are_announced():
    """An excerpt the model does not know is an excerpt is worse than none."""
    env = FakeTree({"src/database/main.py": "a\n"})
    text, meta = sbp.collect_grounding(env, PROBLEM)
    assert "src/user/emails.py" in meta["skipped"]
    assert "not quoted" in text


def test_an_unreadable_container_degrades_to_the_blind_prompt(caplog):
    """Grounding is an enrichment. A task that cannot be read still runs."""
    env = FakeTree(fail=True)
    text, meta = sbp.collect_grounding(env, PROBLEM)
    assert text == ""
    assert "no such container" in meta["error"]
    assert meta["enabled"] is True     # it was asked for, it just could not run


def test_nothing_readable_yields_no_block_rather_than_an_empty_header():
    env = FakeTree({})
    env.grep_paths = lambda terms, limit=40: []
    text, meta = sbp.collect_grounding(env, PROBLEM)
    assert text == "" and meta["read"] == []


# ==============================================================================
# --- WHAT THE PROMPT LOOKS LIKE ---
# ==============================================================================

def test_grounding_is_appended_and_never_replaces_upstreams_three_blocks():
    """The published resolve rates were measured with statement, requirements
    and interface present. Substituting source for any of them would make every
    number here quietly incomparable to the leaderboard."""
    prompt = sbp._solve_prompt(PROBLEM, grounding="--- FILE: x.py ---\nx = 1\n")
    assert PROBLEM["problem_statement"] in prompt
    assert PROBLEM["requirements"] in prompt
    assert PROBLEM["interface"] in prompt
    assert "--- FILE: x.py ---" in prompt


def test_the_repair_prompt_carries_the_same_grounding_as_the_solve_prompt():
    """A repair turn that cannot see the files is asked to fix hunk context it
    still has no way to check."""
    prompt = sbp._repair_prompt(PROBLEM, "diff", "DIGEST",
                                grounding="--- FILE: x.py ---\nx = 1\n")
    assert "--- FILE: x.py ---" in prompt


# ==============================================================================
# --- THE READER ITSELF, WITHOUT DOCKER ---
# ==============================================================================

@pytest.fixture
def env(monkeypatch):
    e = SWEBenchProEnv(PROBLEM)
    e.started = True
    e.shell = []

    def fake_sh(script, timeout=None, check=True):
        e.shell.append(script)
        if script.startswith("git show"):
            path = script.split(":", 1)[1].rstrip("'")
            body = {"a.py": "A" * 50, "big.py": "B" * 5000,
                    "c.py": "C" * 50}.get(path)
            if body is None:
                return types.SimpleNamespace(returncode=128, stdout="", stderr="bad object")
            return types.SimpleNamespace(returncode=0, stdout=body, stderr="")
        if script.startswith("git grep"):
            return types.SimpleNamespace(
                returncode=0, stdout=f"{PROBLEM['base_commit']}:src/hit.py\n", stderr="")
        if script.startswith("git ls-tree"):
            return types.SimpleNamespace(returncode=0, stdout="a.py\nb.py\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(e, "_sh", fake_sh)
    return e


def test_source_is_read_from_the_base_commit_not_the_worktree(env):
    """The container is reused across a task's attempts and the worktree holds
    whatever the last attempt applied. Reading it would show attempt N+1 the
    patch attempt N wrote, which is a leak dressed up as context."""
    env.read_source(["a.py"])
    assert env.shell == [f"git show '{PROBLEM['base_commit']}:a.py'"]


def test_a_path_that_does_not_exist_is_skipped_not_fatal(env):
    blocks, read, skipped = env.read_source(["a.py", "nope.py"])
    assert read == ["a.py"] and skipped == ["nope.py"] and len(blocks) == 1


def test_per_file_cap_clips_and_says_so(env):
    blocks, read, _ = env.read_source(["big.py"], per_file=100)
    assert read == ["big.py"]
    assert "first 100 of 5000 chars" in blocks[0]
    assert len(blocks[0]) < 400


def test_the_total_budget_stops_reading_and_the_rest_is_reported_skipped(env):
    blocks, read, skipped = env.read_source(["a.py", "c.py"], budget=50, per_file=50)
    assert read == ["a.py"] and skipped == ["c.py"]


def test_the_file_count_cap_is_independent_of_the_character_budget(env):
    blocks, read, skipped = env.read_source(["a.py", "c.py"], budget=10 ** 6,
                                            max_files=1)
    assert read == ["a.py"] and skipped == ["c.py"]


def test_grep_strips_the_revision_prefix_git_adds(env):
    """`git grep -l <rev>` prints `<rev>:<path>`; forwarding that verbatim would
    ask `git show` for `<base>:<base>:<path>` and read nothing, silently."""
    assert env.grep_paths(["mget"]) == ["src/hit.py"]


def test_grep_ignores_terms_too_short_to_localise_anything(env):
    assert env.grep_paths(["ab"]) == []
    assert env.shell == []


def test_grep_quotes_terms_containing_a_single_quote(env):
    """A term out of an issue body is untrusted input to a shell command."""
    env.grep_paths(["it's"])
    assert "'it'\\''s'" in env.shell[0]
