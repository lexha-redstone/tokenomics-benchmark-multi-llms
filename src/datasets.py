# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Unified Dataset Loader for BigCodeBench-Hard, WebDev, ClassEval and FeatureBench.
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

# nltk needs data as well as the package, and the resource IDS MOVED between
# releases: `pos_tag` wanted `averaged_perceptron_tagger` before nltk 3.9 and
# `averaged_perceptron_tagger_eng` after it, and `word_tokenize` moved from
# `punkt` to `punkt_tab`. Any hardcoded list is therefore wrong on some
# installed version, whichever list you pick -- documenting the pre-3.9 names
# here once already sent a reader to download data that would not satisfy the
# call.
#
# So the operations the dataset actually performs are probed instead, and the
# installed nltk is asked what IT wants. That cannot drift.
CLASSEVAL_NLTK_PROBES = (
    ("word_tokenize", lambda nltk: nltk.word_tokenize("The cats are running")),
    ("pos_tag", lambda nltk: nltk.pos_tag(["The", "cats", "are", "running"])),
    ("WordNetLemmatizer",
     lambda nltk: nltk.stem.WordNetLemmatizer().lemmatize("running", pos="v")),
)

_NLTK_ATTEMPTED_RE = re.compile(r"Attempted to load\s+'([^']+)'")
_NLTK_RESOURCE_RE = re.compile(r"Resource\s+\W*([\w.-]+)\W*\s+not found")


def classeval_nltk_gaps():
    """nltk data the installed version needs for ClassEval but cannot find.

    Returns a list of dicts with `resource`, `collection`, `path` and
    `needed_by`. Empty when nltk is absent -- a missing package is reported by
    :func:`classeval_missing_modules`, and reporting it twice as two different
    problems helps nobody.
    """
    try:
        import nltk
        import nltk.stem  # noqa: F401  (lazily imported by nltk itself)
    except Exception:
        return []

    gaps, seen = [], set()
    for label, probe in CLASSEVAL_NLTK_PROBES:
        try:
            probe(nltk)
            continue
        except LookupError as exc:
            text = str(exc)
        except Exception:
            continue          # not a data problem; the gold run will surface it

        attempted = _NLTK_ATTEMPTED_RE.search(text)
        resource = _NLTK_RESOURCE_RE.search(text)
        path = (attempted.group(1).strip("/") if attempted else "")
        rid = resource.group(1) if resource else (path.split("/")[-1] if path else "")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        gaps.append({
            "resource": rid,
            "collection": path.split("/")[0] if "/" in path else "",
            "path": path,
            "needed_by": label,
        })
    return gaps


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
# --- FEATUREBENCH DATASET LOADER ---
# ==============================================================================

FEATUREBENCH_DATASET = "LiberCoders/FeatureBench"
FEATUREBENCH_CONFIG = "default"
FEATUREBENCH_DEFAULT_SPLIT = "fast"

# The row fields this repository uses. `problem_statement` runs to 77k chars on
# the largest rows, which is the P4 property FeatureBench was adopted for -- so
# it is kept whole and truncated at prompt-build time, where the budget is
# visible, rather than silently here.
_FB_KEEP_FIELDS = (
    "instance_id", "repo", "base_commit", "problem_statement",
    "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
    "image_name", "repo_settings",
)


def featurebench_quarantine_path(split=FEATUREBENCH_DEFAULT_SPLIT):
    """Where `tools/featurebench_preflight.py` records rows gold cannot pass."""
    return os.path.join(ROOT_DIR, "featurebench", "data", f"quarantine-{split}.json")


def load_featurebench_quarantine(split=FEATUREBENCH_DEFAULT_SPLIT):
    path = featurebench_quarantine_path(split)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("tasks", {})
    except Exception:
        return {}


def ensure_featurebench_dataset(split=FEATUREBENCH_DEFAULT_SPLIT):
    """Ensure the FeatureBench split is present locally, fetching from HF if needed."""
    data_dir = os.path.join(ROOT_DIR, "featurebench", "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"FeatureBench-{split}.jsonl")
    if os.path.exists(path):
        return path

    print(f"Fetching {FEATUREBENCH_DATASET} [{split}] from HuggingFace -> {_rel(path)}",
          flush=True)
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        q = urllib.parse.urlencode({
            "dataset": FEATUREBENCH_DATASET, "config": FEATUREBENCH_CONFIG,
            "split": split, "offset": offset, "length": 20,
        })
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q,
                                    timeout=180, context=_ssl_ctx()) as r:
            d = json.loads(r.read())
        batch = d.get("rows", [])
        total = d.get("num_rows_total", len(batch))
        if not batch:
            break
        rows.extend(b["row"] for b in batch)
        offset += len(batch)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: row.get(k) for k in _FB_KEEP_FIELDS}) + "\n")
    print(f"  saved {len(rows)} instances to {_rel(path)}", flush=True)
    return path


def _fb_repo_settings(raw):
    """`repo_settings` ships as a JSON *string*. Parse it, never guess its keys.

    The preflight prints the keys it actually finds
    (`tools/featurebench_preflight.py --settings`), because binding a wrong key
    here would silently change what every arm is scored on.
    """
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _fb_workdir(settings, repo):
    """Where the repository lives inside its image.

    Read from `repo_settings` when it says so; otherwise `/workspace/<name>`,
    which is what the published images use. The preflight fails loudly on gold
    if this is wrong for a row, so a bad guess cannot reach a scored arm.
    """
    for key in ("workdir", "work_dir", "repo_dir", "repo_path", "root", "cwd"):
        v = settings.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    name = str(repo or "").split("/")[-1] or "repo"
    return f"/workspace/{name}"


def load_featurebench_problems(split=FEATUREBENCH_DEFAULT_SPLIT, max_tasks=None,
                               apply_quarantine=True):
    """Load FeatureBench instances as a dictionary keyed by instance_id.

    Each problem carries the fields the container executor needs -- `image_name`,
    `base_commit`, `test_patch`, `FAIL_TO_PASS`/`PASS_TO_PASS` -- plus a parsed
    `repo_settings` and the `repo_workdir` derived from it.

    Rows whose own gold patch cannot be scored in this environment are excluded
    using the file `tools/featurebench_preflight.py` writes: a missing image, a
    test_patch that will not apply, an image whose pytest cannot collect. As
    with ClassEval, that file is environment-specific -- regenerate it per
    machine rather than copying it.
    """
    path = ensure_featurebench_dataset(split)
    excluded = load_featurebench_quarantine(split) if apply_quarantine else {}
    skipped = 0
    problems = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row.get("instance_id")
            if iid in excluded:
                skipped += 1
                continue
            row["dataset_type"] = "featurebench"
            row["task_id"] = iid
            settings = _fb_repo_settings(row.get("repo_settings"))
            row["settings"] = settings
            row["repo_workdir"] = _fb_workdir(settings, row.get("repo"))
            for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                v = row.get(key) or []
                row[key] = [v] if isinstance(v, str) else list(v)
            problems[iid] = row
            if max_tasks and len(problems) >= max_tasks:
                break
    note = ""
    if skipped:
        note = (f" ({skipped} excluded by "
                f"{os.path.basename(featurebench_quarantine_path(split))}"
                " -- gold does not pass here)")
    elif apply_quarantine and not os.path.exists(featurebench_quarantine_path(split)):
        note = "  [no preflight run yet: python3 tools/featurebench_preflight.py --write]"
    print(f"Loaded {len(problems)} FeatureBench instances from {_rel(path)}{note}")
    return problems


# ==============================================================================
# --- UNIFIED DATASET DISPATCHER ---
# ==============================================================================

def load_dataset(dataset_name, split=None, max_tasks=None):
    """
    Unified dataset dispatcher.
    Supports:
      - 'bcb', 'bigcodebench', 'bigcodebench-hard'
      - 'webdev', 'web-dev'
      - 'classeval', 'class-eval'
      - 'featurebench', 'feature-bench', 'fb'
    """
    name = dataset_name.lower().replace("-", "_")
    if name in ("bcb", "bigcodebench", "bigcodebench_hard"):
        s = split or BCB_DEFAULT_SPLIT
        return load_bcb_problems(split=s, max_tasks=max_tasks)
    elif name in ("webdev", "web_dev"):
        return load_webdev_problems(max_tasks=max_tasks)
    elif name in ("classeval", "class_eval", "ce"):
        s = split or CLASSEVAL_DEFAULT_SPLIT
        return load_classeval_problems(split=s, max_tasks=max_tasks)
    elif name in ("featurebench", "feature_bench", "fb"):
        s = split or FEATUREBENCH_DEFAULT_SPLIT
        return load_featurebench_problems(split=s, max_tasks=max_tasks)
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. "
                         "Supported: 'bcb', 'webdev', 'classeval', 'featurebench'.")
