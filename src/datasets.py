# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Unified Dataset Loader for BigCodeBench-Hard, SWE-bench Pro, and WebDev Benchmarks.
Handles dataset loading, HuggingFace dataset caching, and problem dictionary creation.
"""

import os
import sys
import json
import ast
import importlib.util
import re
import urllib.request
import urllib.parse
import urllib.error

from .paths import display as _rel

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_HERE)

# ==============================================================================
# --- BIGCODEBENCH-HARD DATASET LOADER ---
# ==============================================================================

BCB_DATASET = "bigcode/bigcodebench-hard"
BCB_CONFIG = "default"
BCB_DEFAULT_SPLIT = "v0.1.4"
_BCB_KEEP_FIELDS = ("task_id", "complete_prompt", "canonical_solution", "code_prompt",
                    "test", "entry_point", "libs")

def _ssl_ctx():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def ensure_bcb_dataset(split=BCB_DEFAULT_SPLIT):
    """Ensure BigCodeBench-Hard split file is present locally, fetching from HF if needed."""
    data_dir = os.path.join(ROOT_DIR, "bigCodeBench-hard", "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"BigCodeBench-Hard-{split}.jsonl")
    if os.path.exists(path):
        return path

    print(f"Fetching {BCB_DATASET} [{split}] from HuggingFace -> {_rel(path)}", flush=True)
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        q = urllib.parse.urlencode({
            "dataset": BCB_DATASET, "config": BCB_CONFIG,
            "split": split, "offset": offset, "length": 100
        })
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q,
                                    timeout=120, context=_ssl_ctx()) as r:
            d = json.loads(r.read())
        batch = d.get("rows", [])
        total = d.get("num_rows_total", len(batch))
        if not batch:
            break
        rows.extend(b["row"] for b in batch)
        offset += len(batch)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: row.get(k) for k in _BCB_KEEP_FIELDS}) + "\n")
    print(f"  saved {len(rows)} tasks to {_rel(path)}", flush=True)
    return path

def load_bcb_problems(split=BCB_DEFAULT_SPLIT, max_tasks=None):
    """Load BigCodeBench-Hard tasks as a dictionary keyed by task_id."""
    path = ensure_bcb_dataset(split)
    problems = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["dataset_type"] = "bcb"
                problems[row["task_id"]] = row
                if max_tasks and len(problems) >= max_tasks:
                    break
    return problems

# ==============================================================================
# --- SWE-BENCH PRO DATASET LOADER ---
# ==============================================================================

SWEBENCH_DATASET = "SWE-bench/SWE-bench_Pro"
SWEBENCH_PUBLIC_FILE = "SWE-bench_Pro-public-test.jsonl"

def load_swebench_pro_problems(split="test", max_tasks=None):
    """Load SWE-bench Pro (Public dataset) tasks as a dictionary keyed by instance_id."""
    data_dir = os.path.join(ROOT_DIR, "swebench_pro", "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, SWEBENCH_PUBLIC_FILE)
    
    if not os.path.exists(path):
        # Check alternative locations
        for alt in [
            os.path.join(ROOT_DIR, "swebench-pro", "data", SWEBENCH_PUBLIC_FILE),
            os.path.join(ROOT_DIR, "data", SWEBENCH_PUBLIC_FILE),
        ]:
            if os.path.exists(alt):
                path = alt
                break

    problems = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["dataset_type"] = "swebench"
                    iid = row.get("instance_id") or row.get("task_id")
                    problems[iid] = row
                    if max_tasks and len(problems) >= max_tasks:
                        break
        print(f"Loaded {len(problems)} SWE-bench Pro tasks from {_rel(path)}")
    else:
        print(f"[Notice] SWE-bench Pro dataset file not found at {_rel(path)}.")
        
    return problems

# ==============================================================================
# --- WEB-DEV DATASET LOADER ---
# ==============================================================================

def load_webdev_problems(max_tasks=None):
    """Load Web-Dev tasks filtered by web/networking libraries from BigCodeBench-Hard."""
    web_libs = {
        "requests", "urllib", "flask", "flask_login", "flask_mail", "flask_wtf", 
        "werkzeug", "wtforms", "http", "ftplib", "smtplib", "bs4", "pyquery", "lxml", 
        "cgi", "socket"
    }
    
    path = os.path.join(ROOT_DIR, "webdev", "data", "BigCodeBench-Hard-WebDev.jsonl")
    if not os.path.exists(path):
        # Generate from BCB dataset if needed
        bcb_path = ensure_bcb_dataset(BCB_DEFAULT_SPLIT)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        web_rows = []
        with open(bcb_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                try:
                    lib_list = ast.literal_eval(d.get("libs", "[]"))
                    if any(lib in web_libs for lib in lib_list):
                        web_rows.append(d)
                except Exception:
                    pass
        with open(path, "w", encoding="utf-8") as f:
            for row in web_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Filtered and saved {len(web_rows)} Web-Dev tasks to {_rel(path)}")

    problems = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["dataset_type"] = "webdev"
                problems[d["task_id"]] = d
                if max_tasks and len(problems) >= max_tasks:
                    break
    print(f"Loaded {len(problems)} Web-Dev tasks from {_rel(path)}")
    return problems

# ==============================================================================
# --- CLASSEVAL DATASET LOADER ---
# ==============================================================================
#
# ClassEval is here for one reason BigCodeBench-Hard cannot serve: a task is a
# CLASS, not a function, so it decomposes into several methods whose difficulty
# differs, and every method ships its own test class. That is what makes
# per-sub-task attribution possible -- see docs/pattern-dataset-selection.md.
#
# The difficulty tier is NOT inferred here. It is read off the dataset's own
# `dependencies` annotation, so a routing result can be reported against a label
# the benchmark authors assigned rather than one this repository invented.

CLASSEVAL_DATASET = "FudanSELab/ClassEval"
CLASSEVAL_CONFIG = "default"
CLASSEVAL_DEFAULT_SPLIT = "test"
_CE_KEEP_FIELDS = ("task_id", "skeleton", "test", "solution_code", "import_statement",
                   "class_name", "class_description", "class_constructor", "fields",
                   "methods_info", "test_classes")

# Ordered most-constrained first: a method that calls another method is the
# hardest thing in the class regardless of what else it touches, because it can
# only be written correctly once its callee's contract is settled.
CLASSEVAL_TIERS = ("standalone", "method_dep", "field_lib", "lib_dep", "field_dep")

# Rank is what a difficulty router sorts on. Two tiers share rank 1 because the
# dataset gives no basis for separating them.
CLASSEVAL_TIER_RANK = {"standalone": 0, "lib_dep": 1, "field_dep": 1,
                       "field_lib": 2, "method_dep": 3}


def classeval_tier(dependencies):
    """Map one method's `dependencies` annotation onto a difficulty tier."""
    d = dependencies or {}
    if d.get("Standalone"):
        return "standalone"
    if d.get("method_dependencies"):
        return "method_dep"
    if d.get("field_dependencies") and d.get("lib_dependencies"):
        return "field_lib"
    if d.get("lib_dependencies"):
        return "lib_dep"
    if d.get("field_dependencies"):
        return "field_dep"
    return "field_dep"


# ClassEval's tasks import third-party packages, and a package that is not
# installed does not make the task unscorable -- it makes the MACHINE
# unscorable. That distinction matters more here than it looks: quarantining
# those tasks silently shrinks the benchmark, and two machines with different
# packages installed then measure different task sets, so their numbers cannot
# be compared at all. Ten packages cover every task; installing them is the fix,
# and dropping the tasks is not.
#
# import name -> pip name, for the ones that differ.
CLASSEVAL_PIP_NAMES = {
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "docx": "python-docx",
}

# nltk needs corpora as well as the package; see classeval/requirements.txt.
CLASSEVAL_NLTK_CORPORA = ("punkt", "averaged_perceptron_tagger", "wordnet", "omw-1.4")


def classeval_required_modules(problems=None, split=CLASSEVAL_DEFAULT_SPLIT):
    """Third-party top-level modules the dataset imports, mapped to task ids.

    Read out of the data rather than hardcoded, so a refreshed split cannot
    quietly need something this list does not mention.
    """
    if problems is None:
        problems = load_classeval_problems(split=split, apply_quarantine=False)
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    found = {}
    for prob in problems.values():
        text = "\n".join([
            "\n".join(prob.get("import_statement") or []),
            prob.get("test", "") or "",
            prob.get("solution_code", "") or "",
        ])
        for m in re.finditer(
                r"^\s*(?:import\s+([A-Za-z_][\w.]*)|from\s+([A-Za-z_][\w.]*)\s+import)",
                text, re.M):
            name = (m.group(1) or m.group(2)).split(".")[0]
            if name in stdlib or name == "__future__":
                continue
            found.setdefault(name, set()).add(prob["task_id"])
    return {k: sorted(v) for k, v in sorted(found.items(),
                                            key=lambda kv: (-len(kv[1]), kv[0]))}


def classeval_missing_modules(problems=None, split=CLASSEVAL_DEFAULT_SPLIT):
    """Of those, the ones this interpreter cannot import. Empty is the goal."""
    required = classeval_required_modules(problems, split=split)
    return {name: tasks for name, tasks in required.items()
            if importlib.util.find_spec(name) is None}


def classeval_install_hint(missing):
    """The exact command that closes the gap, or "" when there is none."""
    if not missing:
        return ""
    pkgs = sorted({CLASSEVAL_PIP_NAMES.get(name, name) for name in missing})
    return "pip install " + " ".join(pkgs)


def ensure_classeval_dataset(split=CLASSEVAL_DEFAULT_SPLIT):
    """Ensure the ClassEval split is present locally, fetching from HF if needed."""
    data_dir = os.path.join(ROOT_DIR, "classeval", "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"ClassEval-{split}.jsonl")
    if os.path.exists(path):
        return path

    print(f"Fetching {CLASSEVAL_DATASET} [{split}] from HuggingFace -> {_rel(path)}", flush=True)
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        q = urllib.parse.urlencode({
            "dataset": CLASSEVAL_DATASET, "config": CLASSEVAL_CONFIG,
            "split": split, "offset": offset, "length": 50
        })
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q,
                                    timeout=120, context=_ssl_ctx()) as r:
            d = json.loads(r.read())
        batch = d.get("rows", [])
        total = d.get("num_rows_total", len(batch))
        if not batch:
            break
        rows.extend(b["row"] for b in batch)
        offset += len(batch)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: row.get(k) for k in _CE_KEEP_FIELDS}) + "\n")
    print(f"  saved {len(rows)} classes to {_rel(path)}", flush=True)
    return path


def _classeval_decorators(skeleton):
    """Map method name -> its decorator lines, read off the skeleton.

    `methods_info[*].solution_code` is unreliable here: for a decorated method
    the extractor sometimes keeps `@staticmethod` and sometimes drops it (in
    ClassEval_3, `count_all` keeps it and `count` does not). Assembling a class
    method-by-method from those strings therefore silently loses decorators and
    the class fails with "takes 2 positional arguments but 3 were given" --
    a defect of the assembly, not of the model that wrote the body. The
    skeleton, which is also what the model is shown, always carries them.
    """
    out, pending = {}, []
    for raw in (skeleton or "").splitlines():
        line = raw.strip()
        if line.startswith("@"):
            pending.append(line)
            continue
        m = re.match(r"def\s+([A-Za-z_]\w*)\s*\(", line)
        if m:
            if pending:
                out[m.group(1)] = list(pending)
            pending = []
        elif line:
            pending = []
    return out


def _classeval_subtasks(row):
    """Normalise `methods_info` into the sub-task shape the arms and the
    reporter both consume. Keeps the raw annotation alongside the tier so a
    disputed tier can always be re-derived."""
    out = []
    decorators = _classeval_decorators(row.get("skeleton", ""))
    for m in row.get("methods_info") or []:
        deps = m.get("dependencies") or {}
        tier = classeval_tier(deps)
        name = m.get("method_name", "")
        out.append({
            "name": name,
            "decorators": decorators.get(name, []),
            "tier": tier,
            "rank": CLASSEVAL_TIER_RANK.get(tier, 1),
            "description": m.get("method_description", ""),
            "test_class": str(m.get("test_class", "") or "").strip(),
            "test_code": m.get("test_code", ""),
            "solution_code": m.get("solution_code", ""),
            "dependencies": deps,
        })
    return out


def quarantine_path(split=CLASSEVAL_DEFAULT_SPLIT):
    """Where `tools/classeval_preflight.py` records tasks gold cannot pass."""
    return os.path.join(ROOT_DIR, "classeval", "data", f"quarantine-{split}.json")


def load_quarantine(split=CLASSEVAL_DEFAULT_SPLIT):
    """Task ids the environment cannot score, mapped to why. Empty if unrun."""
    path = quarantine_path(split)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("tasks", {})
    except Exception:
        return {}


def load_classeval_problems(split=CLASSEVAL_DEFAULT_SPLIT, max_tasks=None,
                            apply_quarantine=True):
    """Load ClassEval tasks as a dictionary keyed by task_id.

    Each problem carries a `subtasks` list -- one entry per method, with the
    tier, the method's own test class, and that class's source. `integration_tests`
    holds the test classes belonging to no single method; they are the ones that
    only fail when the methods do not compose, which is precisely the failure a
    planner is supposed to prevent.
   
    Tasks whose own gold solution fails in this environment are excluded, using
    the file `tools/classeval_preflight.py` writes. Six of the hundred fail for
    reasons that belong to the machine rather than to any model -- a missing
    optional import, an undownloaded corpus, gold written against NumPy 1.x --
    and scoring an arm against those measures the environment. Pass
    `apply_quarantine=False` to see the raw set (which is what the preflight
    itself must do).
    """
    path = ensure_classeval_dataset(split)
    excluded = load_quarantine(split) if apply_quarantine else {}
    skipped = 0
    problems = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") in excluded:
                skipped += 1
                continue
            row["dataset_type"] = "classeval"
            imports = row.get("import_statement") or []
            row["import_block"] = "\n".join(imports) if isinstance(imports, list) else str(imports)
            row["subtasks"] = _classeval_subtasks(row)
            # `test_classes` carries stray whitespace on at least one row
            # (ClassEval_97 ships " Words2NumbersTestMain"), which the runner
            # tail would then fail to resolve as a name.
            row["test_classes"] = [str(t).strip() for t in (row.get("test_classes") or [])
                                   if str(t).strip()]
            owned = {s["test_class"] for s in row["subtasks"]}
            row["integration_tests"] = [t for t in row["test_classes"] if t not in owned]
            problems[row["task_id"]] = row
            if max_tasks and len(problems) >= max_tasks:
                break
    note = ""
    if skipped:
        note = (f" ({skipped} excluded by {os.path.basename(quarantine_path(split))}"
                " -- gold does not pass here)")
    elif apply_quarantine and not os.path.exists(quarantine_path(split)):
        note = "  [no preflight run yet: python3 tools/classeval_preflight.py --write]"
    print(f"Loaded {len(problems)} ClassEval tasks from {_rel(path)}{note}")
    return problems


# ==============================================================================
# --- UNIFIED DATASET DISPATCHER ---
# ==============================================================================

def load_dataset(dataset_name, split=None, max_tasks=None):
    """
    Unified dataset dispatcher.
    Supports:
      - 'bcb', 'bigcodebench', 'bigcodebench-hard'
      - 'swebench', 'swebench_pro', 'swe-bench'
      - 'webdev', 'web-dev'
      - 'classeval', 'class-eval'
    """
    name = dataset_name.lower().replace("-", "_")
    if name in ("bcb", "bigcodebench", "bigcodebench_hard"):
        s = split or BCB_DEFAULT_SPLIT
        return load_bcb_problems(split=s, max_tasks=max_tasks)
    elif name in ("swebench", "swebench_pro", "swe_bench", "swe_bench_pro"):
        s = split or "test"
        return load_swebench_pro_problems(split=s, max_tasks=max_tasks)
    elif name in ("webdev", "web_dev"):
        return load_webdev_problems(max_tasks=max_tasks)
    elif name in ("classeval", "class_eval", "ce"):
        s = split or CLASSEVAL_DEFAULT_SPLIT
        return load_classeval_problems(split=s, max_tasks=max_tasks)
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. "
                         "Supported: 'bcb', 'swebench', 'webdev', 'classeval'.")
