# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Contract tests for the ClassEval arms (`src/classeval.py`).

ClassEval is in this repository to answer one question: does routing sub-tasks
by difficulty beat a cascade? That answer is only worth anything if the plumbing
underneath it is honest, so these pin the three things that would silently
corrupt the measurement:

  * **assembly.** The per-method arms build a class out of separately generated
    methods. If assembly loses a `@staticmethod`, or mis-indents a body, the
    class fails for a reason that has nothing to do with the model -- and it
    fails only for the per-method arms, which is a bias between the very arms
    being compared. The round-trip test (gold methods in, passing class out) is
    the guard.
  * **tiering.** The difficulty label must come from the dataset's own
    `dependencies` annotation. If this repo silently re-derived it, a routing
    result would be reported against a label of its own invention.
  * **selection.** The runner tail must run exactly the test classes it was
    given, and must fail loudly rather than obscurely when a candidate never
    defined one.

No test here calls a model.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import straitjacket as sj  # noqa: E402
from src.classeval import (CLASSEVAL_VARIANTS, LADDER, TIER_ROUTE, assemble_class,
                           extract_method, normalise_method)  # noqa: E402
from src.datasets import (classeval_tier, load_classeval_problems,
                          quarantine_path)  # noqa: E402
from src.evaluator import (missing_class_error, run_classeval_class,
                           run_classeval_method,
                           classeval_subtask_summary)  # noqa: E402

pytestmark = pytest.mark.skipif(
    not sj.available(),
    reason="ctx-harness not installed (pip install ctx-harness)",
)

# A handful is enough for the contract; the full sweep is the preflight's job.
SAMPLE = 12


@pytest.fixture(scope="module")
def problems():
    p = load_classeval_problems(max_tasks=SAMPLE)
    if not p:
        pytest.skip("ClassEval data not available")
    return p


# ==============================================================================
# --- TIER LABELS COME FROM THE DATASET ---
# ==============================================================================

@pytest.mark.parametrize("deps,expected", [
    ({"Standalone": True, "lib_dependencies": [], "field_dependencies": [],
      "method_dependencies": []}, "standalone"),
    ({"Standalone": False, "method_dependencies": ["other"],
      "lib_dependencies": ["os"], "field_dependencies": ["x"]}, "method_dep"),
    ({"Standalone": False, "method_dependencies": [],
      "lib_dependencies": ["datetime"], "field_dependencies": ["x"]}, "field_lib"),
    ({"Standalone": False, "method_dependencies": [],
      "lib_dependencies": ["datetime"], "field_dependencies": []}, "lib_dep"),
    ({"Standalone": False, "method_dependencies": [],
      "lib_dependencies": [], "field_dependencies": ["x"]}, "field_dep"),
])
def test_tier_precedence(deps, expected):
    """A method that calls another method is the hardest thing in the class
    whatever else it touches, so `method_dep` outranks every other flag."""
    assert classeval_tier(deps) == expected


def test_every_tier_has_a_route():
    """A tier with no entry in TIER_ROUTE would silently take the fallback,
    and the arm would be routing by accident rather than by policy."""
    for tier in ("standalone", "lib_dep", "field_dep", "field_lib", "method_dep"):
        assert tier in TIER_ROUTE
        model, _think = TIER_ROUTE[tier]
        assert model in LADDER, f"{tier} routes off the ladder"


def test_route_is_monotone_in_difficulty():
    """Harder tiers must not be routed to a cheaper rung than easier ones."""
    rank_of = {"standalone": 0, "lib_dep": 1, "field_dep": 1,
               "field_lib": 2, "method_dep": 3}
    for a, ra in rank_of.items():
        for b, rb in rank_of.items():
            if ra < rb:
                assert LADDER.index(TIER_ROUTE[a][0]) <= LADDER.index(TIER_ROUTE[b][0])


def test_subtasks_carry_tier_and_test_class(problems):
    for prob in problems.values():
        assert prob["subtasks"], f"{prob['task_id']} has no sub-tasks"
        for sub in prob["subtasks"]:
            assert sub["tier"] in TIER_ROUTE
            assert sub["test_class"], f"{prob['task_id']}.{sub['name']} has no test class"
            assert sub["test_class"] == sub["test_class"].strip()
            assert sub["test_class"] in prob["test_classes"]


# ==============================================================================
# --- ASSEMBLY ---
# ==============================================================================

def test_normalise_dataset_shape():
    """Dataset shape: first line dedented to 0, body still at 8."""
    out = normalise_method("def f(self, x):\n        return x + 1")
    assert out == "    def f(self, x):\n        return x + 1"


def test_normalise_model_shape():
    """Model shape: a self-consistent block, body at 4."""
    out = normalise_method("def f(self, x):\n    return x + 1")
    assert out == "    def f(self, x):\n        return x + 1"


def test_normalise_keeps_decorator_alignment():
    """`@staticmethod` at column 0 with its def already at 4 must not end up
    with the decorator and the def on different indents."""
    out = normalise_method("@staticmethod\n    def f(x):\n        return x")
    assert out == "    @staticmethod\n    def f(x):\n        return x"


def test_normalise_restores_dropped_decorator():
    """`methods_info` drops `@staticmethod` on some rows. A lost decorator
    surfaces as an argument-count error that reads exactly like a model
    mistake, so it is restored from the skeleton."""
    out = normalise_method("def f(x):\n    return x", decorators=["@staticmethod"])
    assert out.startswith("    @staticmethod\n    def f(x):")


def test_normalise_does_not_double_a_decorator():
    out = normalise_method("@staticmethod\n    def f(x):\n        return x",
                           decorators=["@staticmethod"])
    assert out.count("@staticmethod") == 1


def test_extract_method_from_fenced_block():
    text = "```python\ndef only(self):\n    return 1\n```"
    assert extract_method(text, "only").strip().startswith("def only")


def test_extract_method_picks_the_named_one():
    text = "```python\ndef a(self):\n    return 1\n\ndef b(self):\n    return 2\n```"
    got = extract_method(text, "b")
    assert "def b" in got and "def a" not in got


def test_extract_method_keeps_decorator():
    text = "```python\n@staticmethod\ndef a(x):\n    return x\n```"
    assert extract_method(text, "a").lstrip().startswith("@staticmethod")


def test_assembly_round_trip(problems):
    """The load-bearing test: a class rebuilt from its own gold methods must
    still pass its own tests. Anything less means the per-method arms are being
    scored on assembly defects."""
    bad = []
    for prob in problems.values():
        sources = {s["name"]: s["solution_code"] for s in prob["subtasks"]}
        passed, ev = run_classeval_class(prob, assemble_class(prob, sources))
        if not passed:
            bad.append((prob["task_id"], str(ev)[:120]))
    assert not bad, f"assembly broke gold classes: {bad}"


def test_missing_method_becomes_a_stub_not_a_syntax_error(problems):
    """A model that skipped one method must lose that method's tests, not take
    the whole class down and charge the failure to every other method."""
    prob = next(iter(problems.values()))
    subs = prob["subtasks"]
    if len(subs) < 2:
        pytest.skip("needs a class with at least two methods")
    sources = {s["name"]: s["solution_code"] for s in subs[1:]}
    code = assemble_class(prob, sources)
    compile(code, "<assembled>", "exec")          # must at least parse
    kept_ok, _ = run_classeval_method(prob, code, subs[1])
    dropped_ok, _ = run_classeval_method(prob, code, subs[0])
    assert kept_ok, "a surviving method should still pass"
    assert not dropped_ok, "the omitted method should fail its own tests"


# ==============================================================================
# --- SCORING AND SELECTION ---
# ==============================================================================

def test_gold_passes_class_and_every_method(problems):
    for prob in problems.values():
        assert run_classeval_class(prob, prob["solution_code"])[0], prob["task_id"]
        for sub in prob["subtasks"]:
            ok, _ = run_classeval_method(prob, prob["solution_code"], sub)
            assert ok, f"{prob['task_id']}.{sub['name']}"


def test_broken_method_fails_only_its_own_tests(problems):
    """Per-method scoring has to be independent, or attribution is fiction."""
    prob = next(p for p in problems.values() if len(p["subtasks"]) >= 2)
    target = prob["subtasks"][0]
    sources = {s["name"]: s["solution_code"] for s in prob["subtasks"]}
    sources[target["name"]] = (f"def {target['name']}(self, *args, **kwargs):\n"
                               f"    return None")
    code = assemble_class(prob, sources)
    assert not run_classeval_method(prob, code, target)[0]
    others = [s for s in prob["subtasks"][1:]
              if run_classeval_method(prob, code, s)[0]]
    assert others, "breaking one method should leave at least one other passing"


def test_tail_runs_only_the_named_class():
    tail = sj.unittest_tail(["Wanted"])
    assert "'Wanted'" in tail and "Unwanted" not in tail


def test_tail_reports_a_missing_class_instead_of_NameError(tmp_path):
    prog = "import unittest\n" + sj.unittest_tail(["Nope"])
    path = tmp_path / "p.py"
    path.write_text(prog, encoding="utf-8")
    import subprocess
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "MissingTestClass" in r.stderr
    assert "NameError" not in r.stderr


def test_bcb_tail_is_untouched():
    """DETERMINISTIC_UNITTEST_TAIL's bytes are part of BCB's capture identity;
    editing it would re-mint every stored artifact."""
    assert "loadTestsFromTestCase(TestCases)" in sj.DETERMINISTIC_UNITTEST_TAIL


def test_missing_class_gate():
    assert missing_class_error("def f(): pass", "Widget") is not None
    assert missing_class_error("class Widget:\n    pass", "Widget") is None


def test_subtask_summary_splits_by_tier_and_model():
    records = [
        {"name": "a", "tier": "standalone", "model_id": "cheap", "passed": True,
         "as_run_usd": 0.001},
        {"name": "b", "tier": "standalone", "model_id": "cheap", "passed": False,
         "as_run_usd": 0.001},
        {"name": "c", "tier": "method_dep", "model_id": "strong", "passed": True,
         "as_run_usd": 0.01},
    ]
    s = classeval_subtask_summary(records)
    assert s["n_subtasks"] == 3 and s["passed_subtasks"] == 2
    assert s["by_tier"]["standalone"]["pass_rate"] == 0.5
    assert s["by_tier"]["method_dep"]["pass_rate"] == 1.0
    assert s["by_model"]["cheap"]["n"] == 2
    assert s["by_model"]["strong"]["usd"] == 0.01


def test_whole_class_spend_is_split_not_zeroed():
    """A whole-class arm buys its methods as a bundle. Recording 0.00 per
    method would make the by-tier rollup read as if those methods were free,
    and the cost comparison H1 turns on would favour the cascade by accident."""
    from src.classeval import _result
    records = [{"name": "a", "tier": "standalone", "rank": 0, "passed": True},
               {"name": "b", "tier": "method_dep", "rank": 3, "passed": False}]
    out = _result({}, False, "err", records, usd=0.10, out_tok=10, tot_tok=20, loops=1)
    costs = [r["as_run_usd"] for r in out["subtasks"]]
    assert costs == [0.05, 0.05]
    assert all(r["cost_basis"].startswith("class-level") for r in out["subtasks"])
    assert sum(costs) == pytest.approx(out["as_run_usd"])


def test_per_method_spend_is_not_overwritten():
    from src.classeval import _result
    records = [{"name": "a", "tier": "standalone", "rank": 0, "passed": True},
               {"name": "b", "tier": "method_dep", "rank": 3, "passed": True}]
    out = _result({}, True, "", records, usd=0.11, out_tok=10, tot_tok=20, loops=0,
                  usd_by_method={"a": 0.01, "b": 0.10})
    got = {r["name"]: r["as_run_usd"] for r in out["subtasks"]}
    assert got == {"a": 0.01, "b": 0.10}
    assert not any("cost_basis" in r for r in out["subtasks"])


def test_score_records_first_and_final_writer(problems):
    """A repair turn changes the writer. A rollup that only knew the final one
    would credit the escalation rung with work the cheap rung already did."""
    from src.classeval import _score
    prob = next(iter(problems.values()))
    names = [s["name"] for s in prob["subtasks"]]
    first = {n: "cheap" for n in names}
    final = dict(first)
    final[names[0]] = "strong"
    _passed, _ev, records = _score(prob, prob["solution_code"], final, first)
    by_name = {r["name"]: r for r in records}
    assert by_name[names[0]]["model_id"] == "strong"
    assert by_name[names[0]]["initial_model_id"] == "cheap"
    assert by_name[names[0]]["repaired"] is True
    for n in names[1:]:
        assert by_name[n]["repaired"] is False


# ==============================================================================
# --- QUARANTINE AND WIRING ---
# ==============================================================================

def test_quarantine_is_applied_when_present():
    """Tasks whose own gold solution fails here measure the environment, not
    the model, so they must not reach an arm."""
    if not os.path.exists(quarantine_path()):
        pytest.skip("preflight has not been run (tools/classeval_preflight.py --write)")
    kept = load_classeval_problems(apply_quarantine=True)
    raw = load_classeval_problems(apply_quarantine=False)
    assert len(kept) <= len(raw)
    import json
    with open(quarantine_path(), encoding="utf-8") as f:
        excluded = json.load(f)["tasks"]
    assert not (set(kept) & set(excluded))


# ==============================================================================
# --- ENVIRONMENT, NOT TASK ---
# ==============================================================================

def test_required_modules_are_read_from_the_data():
    """Hardcoding the list would let a refreshed split quietly need something
    the requirements file never mentions."""
    from src.datasets import classeval_required_modules
    req = classeval_required_modules()
    assert "numpy" in req, "numpy is used by several tasks"
    for name, tasks in req.items():
        assert tasks, f"{name} listed with no task"
        assert all(t.startswith("ClassEval_") for t in tasks)
    import sys as _sys
    assert not (set(req) & set(_sys.stdlib_module_names)), "stdlib leaked in"


def test_install_hint_uses_pip_names_not_import_names():
    """`pip install PIL` fails; the package is Pillow. A hint that cannot be
    pasted is worse than none."""
    from src.datasets import classeval_install_hint
    hint = classeval_install_hint({"PIL": ["x"], "bs4": ["y"], "docx": ["z"],
                                   "numpy": ["w"]})
    assert "Pillow" in hint and "beautifulsoup4" in hint and "python-docx" in hint
    assert "numpy" in hint
    assert " PIL" not in hint and "bs4" not in hint


def test_install_hint_is_empty_when_nothing_is_missing():
    from src.datasets import classeval_install_hint
    assert classeval_install_hint({}) == ""


def test_requirements_file_covers_every_required_module():
    """The requirements file and the dataset must not drift apart."""
    from src.datasets import CLASSEVAL_PIP_NAMES, classeval_required_modules
    req_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "classeval", "requirements.txt")
    if not os.path.exists(req_path):
        pytest.skip("classeval/requirements.txt not present")
    with open(req_path, encoding="utf-8") as f:
        listed = {line.split("#")[0].strip().lower()
                  for line in f if line.split("#")[0].strip()}
    for name in classeval_required_modules():
        pip_name = CLASSEVAL_PIP_NAMES.get(name, name).lower()
        assert pip_name in listed, f"{name} ({pip_name}) missing from requirements.txt"


def test_nltk_gaps_are_probed_not_hardcoded():
    """The resource ids moved between nltk releases, so the check has to ask
    the installed nltk what it wants. A hardcoded list sent one reader to
    download `averaged_perceptron_tagger` when `pos_tag` wanted
    `averaged_perceptron_tagger_eng`."""
    from src.datasets import classeval_nltk_gaps
    gaps = classeval_nltk_gaps()
    assert isinstance(gaps, list)
    for g in gaps:
        assert g["resource"], "a gap with no resource id is not actionable"
        assert g["needed_by"] in {"word_tokenize", "pos_tag", "WordNetLemmatizer"}
        assert g["collection"] in {"taggers", "tokenizers", "corpora", ""}


def test_nltk_gaps_is_empty_without_nltk(monkeypatch):
    """A missing package is reported once, by the module check -- not a second
    time as a data problem."""
    import builtins
    from src import datasets as ds
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "nltk" or name.startswith("nltk."):
            raise ImportError("blocked for test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert ds.classeval_nltk_gaps() == []


def test_fetch_tool_builds_the_right_url():
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "fetch_nltk_data", os.path.join(root, "tools", "fetch_nltk_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.BASE.startswith("https://")
    assert mod.FALLBACK_COLLECTION["punkt_tab"] == "tokenizers"
    assert mod.FALLBACK_COLLECTION["averaged_perceptron_tagger_eng"] == "taggers"
    dest = mod.default_dest()
    assert os.path.isabs(dest) or dest.startswith("~")


def test_the_control_arm_exists():
    """A routed arm without its flat control cannot support any claim about
    difficulty routing, so its absence is a test failure, not an omission."""
    assert "ce_route_flat" in CLASSEVAL_VARIANTS
    assert "ce_route_by_tier" in CLASSEVAL_VARIANTS


def test_variants_are_wired_into_the_runner():
    from src.architectures import get_configurations
    ids = {c["id"] for c in get_configurations(dataset="classeval", group="classeval")}
    assert set(CLASSEVAL_VARIANTS) == ids
    # and must not leak into the other datasets' groups
    bcb = {c["id"] for c in get_configurations(dataset="bcb", group="all")}
    assert not (bcb & set(CLASSEVAL_VARIANTS))
