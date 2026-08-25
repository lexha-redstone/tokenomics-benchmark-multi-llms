# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Unified Dataset Loader for BigCodeBench-Hard, WebDev, ClassEval, FeatureBench
and SWE-bench Pro.
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


def _fetch(url, timeout=180, max_retries=6):
    """GET `url` as bytes, with auth, backoff on rate limits, and `curl` fallback for CA-less interpreters.

    A Python installed without a certificate bundle (common on macOS, and the
    state of the interpreter this was written on) fails every HTTPS request
    with CERTIFICATE_VERIFY_FAILED even though the machine itself is online.
    That is an environment problem, not a dataset problem, so it is worked
    around here rather than turned into "the dataset could not be fetched".
    """
    import time as _time
    headers = {"User-Agent": "tokenomics-benchmark/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                retry_after = e.headers.get("Retry-After")
                try:
                    sleep_sec = float(retry_after) if retry_after else min(60.0, 2.0 ** attempt * 2.0)
                except (ValueError, TypeError):
                    sleep_sec = min(60.0, 2.0 ** attempt * 2.0)
                print(f"  [HTTP {e.code}] rate limited by HuggingFace; sleeping {sleep_sec:.1f}s (retry {attempt + 1}/{max_retries})...",
                      flush=True)
                _time.sleep(sleep_sec)
                continue
            raise
        except Exception as e:                                   # noqa: BLE001
            if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                if attempt < max_retries:
                    _time.sleep(min(30.0, 2.0 ** attempt))
                    continue
                raise
            import shutil as _shutil
            import subprocess as _subprocess
            if _shutil.which("curl") is None:
                raise
            curl_cmd = ["curl", "-fsSL", "--max-time", str(int(timeout))]
            for k, v in headers.items():
                curl_cmd.extend(["-H", f"{k}: {v}"])
            curl_cmd.append(url)
            p = _subprocess.run(curl_cmd, capture_output=True, timeout=timeout + 30)
            if p.returncode != 0:
                raise RuntimeError(
                    f"urllib has no CA bundle and curl failed for {url}: "
                    f"{(p.stderr or b'').decode('utf-8', 'replace')[-200:]}") from e
            return p.stdout


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
        d = json.loads(_fetch("https://datasets-server.huggingface.co/rows?" + q))
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
        d = json.loads(_fetch("https://datasets-server.huggingface.co/rows?" + q))
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
        d = json.loads(_fetch("https://datasets-server.huggingface.co/rows?" + q))
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
    """Last-resort guess at where the repository lives inside its image.

    `repo_settings` does **not** carry a workdir -- verified across all 100 rows
    of the fast split, whose keys are `repository`, `commit`, `base_image`,
    `install`, `test_cmd`, ... and nothing path-like. So this is a fallback
    only: `FeatureBenchEnv` resolves the real location at container start from
    the image's own `WORKDIR` and then from the git root, both of which are
    authoritative. Guessing was the original design and the preflight was built
    to catch it; the data said not to guess at all.
    """
    name = str(settings.get("library_name") or "").strip() \
        or str(repo or "").split("/")[-1] or "repo"
    return f"/workspace/{name}"


def fb_test_command(problem):
    """The command that runs a row's tests, from the dataset rather than guessed.

    Every row carries `test_cmd`. Falling back to a hardcoded pytest invocation
    would score the arms on a command the benchmark never specified.
    """
    settings = problem.get("settings") or {}
    cmd = settings.get("test_cmd")
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(c) for c in cmd)
    cmd = str(cmd or "").strip()
    return cmd or "python -m pytest -q --tb=short -p no:cacheprovider"


def fb_timeout(problem, key="timeout_run", default=900.0):
    """Per-row timeout from `repo_settings`, which ships several."""
    v = (problem.get("settings") or {}).get(key)
    try:
        return float(v) if v else float(default)
    except (TypeError, ValueError):
        return float(default)


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
# --- SWE-BENCH PRO DATASET LOADER ---
# ==============================================================================
#
# Why this dataset is here
# ------------------------
# FeatureBench was adopted to test H2 -- does front-loaded planning beat
# fail->escalate once the oracle costs real money -- and too many of its rows
# are unscorable here for a reason that has nothing to do with H2: a row is
# only gradable when the repository's own `test_patch` applies to the image it
# ships with, and the harness has to reconstruct the graded tree itself. An arm
# cannot be measured on a row whose *gold* patch cannot pass.
#
# SWE-bench Pro is the same shape of task -- multi-file work in a real
# repository, graded by that repository's own tests inside a per-instance
# Docker image -- with the fragile step removed. Nothing is reconstructed
# locally, because every row ships the commands upstream itself runs:
#
#   dockerhub_tag              the prebuilt image: repo cloned at /app, deps
#                              installed, ENTRYPOINT ["/bin/bash"]
#   before_repo_set_cmd        the exact git commands that put the graded test
#                              files in place (last line does the checkout)
#   selected_test_files_to_run what the run script is pointed at
#   fail_to_pass/pass_to_pass  test *names* (not file paths, unlike FeatureBench)
#
# and upstream publishes, per instance, a `run_script.sh` (how to run that
# repository's tests) and a `parser.py` (how to turn its output into
# {name, status} records) in scaleapi/SWE-bench_Pro-os. So the grading rule is
# the benchmark's own rather than a reimplementation of it: resolved means
# every required test name came back PASSED, which is exactly what upstream's
# `swe_bench_pro_eval.py` computes.
#
# What this module does NOT do
# ----------------------------
# It does not use Modal. Upstream's default runtime is a Modal sandbox; the
# `--use_local_docker` path is the one mirrored here, because the question this
# repository asks is about routing policy and dollars, and a second scheduler
# in the loop is one more thing that can fail in a way the numbers absorb.

SWEBENCH_PRO_DATASET = "ScaleAI/SWE-bench_Pro"
SWEBENCH_PRO_CONFIG = "default"
SWEBENCH_PRO_DEFAULT_SPLIT = "test"

# The per-instance run scripts and Dockerfiles are not in the HuggingFace
# dataset; they live in the evaluation repository and are fetched on demand.
SWEBENCH_PRO_SCRIPTS_RAW = ("https://raw.githubusercontent.com/"
                            "scaleapi/SWE-bench_Pro-os/main")

# Upstream publishes the images under one account. `dockerhub_tag` is a dataset
# column, so the tag is never derived from the instance_id here -- upstream's
# own `get_dockerhub_image_uri` carries two special cases for element-web that
# a re-derivation would silently get wrong.
SWEBENCH_PRO_DOCKERHUB_USER = os.environ.get("SBP_DOCKERHUB_USER", "jefzda")
SWEBENCH_PRO_IMAGE_REPO = os.environ.get("SBP_IMAGE_REPO", "sweap-images")

# Every field the container executor or an arm's prompt reads. `pass_to_pass`
# runs to 45k chars on the larger rows and is kept whole: it is the grading
# denominator, so truncating it would silently change what "resolved" means.
_SBP_KEEP_FIELDS = (
    "instance_id", "repo", "base_commit", "patch", "test_patch",
    "problem_statement", "requirements", "interface", "repo_language",
    "fail_to_pass", "pass_to_pass", "issue_specificity", "issue_categories",
    "before_repo_set_cmd", "selected_test_files_to_run", "dockerhub_tag",
)

# The list-valued columns arrive as *Python literals*, not JSON: upstream reads
# them with `eval()`, and the sample row's fail_to_pass mixes `"` and `'`
# quoting because one test name contains an apostrophe. `json.loads` fails on
# exactly those rows, which would drop required tests and inflate pass rates.
_SBP_LIST_FIELDS = ("fail_to_pass", "pass_to_pass", "selected_test_files_to_run",
                    "issue_specificity", "issue_categories")


def _sbp_list(raw):
    """Parse one of the dataset's list-valued string columns."""
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    text = str(raw or "").strip()
    if not text:
        return []
    for parse in (ast.literal_eval, json.loads):
        try:
            v = parse(text)
        except Exception:
            continue
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return [str(v)]
    # Not a literal at all: treat it as a single entry rather than dropping it
    # silently, so a schema change shows up as a failing row and not as a
    # smaller grading set.
    return [text]


def swebench_pro_dir(*parts):
    return os.path.join(ROOT_DIR, "swebench_pro", *parts)


def swebench_pro_quarantine_path(split=SWEBENCH_PRO_DEFAULT_SPLIT):
    """Where `tools/swebench_pro_preflight.py` records rows gold cannot pass."""
    return swebench_pro_dir("data", f"quarantine-{split}.json")


def load_swebench_pro_quarantine(split=SWEBENCH_PRO_DEFAULT_SPLIT):
    path = swebench_pro_quarantine_path(split)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("tasks", {})
    except Exception:
        return {}


def ensure_swebench_pro_dataset(split=SWEBENCH_PRO_DEFAULT_SPLIT):
    """Ensure the SWE-bench Pro split is present locally, fetching it if needed."""
    data_dir = swebench_pro_dir("data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"SWE-bench_Pro-{split}.jsonl")
    if os.path.exists(path):
        return path

    print(f"Fetching {SWEBENCH_PRO_DATASET} [{split}] from HuggingFace -> {_rel(path)}",
          flush=True)
    rows, offset, total = [], 0, None
    import time as _time
    while total is None or offset < total:
        q = urllib.parse.urlencode({
            "dataset": SWEBENCH_PRO_DATASET, "config": SWEBENCH_PRO_CONFIG,
            # 20 rows is not a round number, it is the largest page that comes
            # back whole: the rows server truncates oversized cells and says so
            # in `truncated_cells`, and a truncated `pass_to_pass` is a smaller
            # grading set rather than an error.
            "split": split, "offset": offset, "length": 20,
        })
        d = json.loads(_fetch("https://datasets-server.huggingface.co/rows?" + q))
        batch = d.get("rows", [])
        total = d.get("num_rows_total", len(batch))
        if not batch:
            break
        for b in batch:
            if b.get("truncated_cells"):
                raise RuntimeError(
                    f"row {b.get('row_idx')} came back with truncated cells "
                    f"{b['truncated_cells']} -- refusing to grade against a "
                    "partial test list. Re-fetch with a smaller page size.")
        rows.extend(b["row"] for b in batch)
        offset += len(batch)
        print(f"  fetched {len(rows)}/{total or '?'} instances...", end="\r", flush=True)
        _time.sleep(0.15)
    print("", flush=True)

    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: row.get(k) for k in _SBP_KEEP_FIELDS}) + "\n")
    os.replace(tmp, path)
    print(f"  saved {len(rows)} instances to {_rel(path)}", flush=True)
    return path


def swebench_pro_image(problem, username=None, repo=None):
    """The prebuilt image for a row, from the dataset's own `dockerhub_tag`."""
    tag = str(problem.get("dockerhub_tag") or "").strip()
    if not tag:
        return ""
    user = username or SWEBENCH_PRO_DOCKERHUB_USER
    return f"{user}/{repo or SWEBENCH_PRO_IMAGE_REPO}:{tag}"


def swebench_pro_scripts_dir(instance_id):
    return swebench_pro_dir("run_scripts", str(instance_id))


def _sbp_cached(url, path, refresh=False):
    if not refresh and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = _fetch(url).decode("utf-8", "replace")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def _sbp_env_exports(instance_id, refresh=False):
    """The image's own ENV lines, replayed as `export` statements.

    Belt and braces, and upstream does the same: `docker exec` already inherits
    the image's environment, but a run script that assumes `$SETUP` or
    `$PYTEST_ADDOPTS` is cheap to protect and expensive to debug. Only the
    `ENV k=v` form is converted -- upstream's blanket `ENV` -> `export`
    rewrite turns the legacy `ENV k v` form into a command that fails.
    """
    out = []
    for kind in ("base_dockerfile", "instance_dockerfile"):
        url = f"{SWEBENCH_PRO_SCRIPTS_RAW}/dockerfiles/{kind}/{instance_id}/Dockerfile"
        path = swebench_pro_scripts_dir(instance_id) + f"/{kind}.Dockerfile"
        try:
            text = _sbp_cached(url, path, refresh=refresh)
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^ENV\s+[A-Za-z_][A-Za-z_0-9]*=", line):
                out.append("export " + line[3:].strip())
    return "\n".join(out)


def ensure_swebench_pro_scripts(instance_id, refresh=False):
    """Fetch (and cache) the per-instance run script, parser and ENV exports.

    These are the benchmark's own grading machinery. Reimplementing either one
    would mean scoring the arms on a test command and a log parser that
    SWE-bench Pro never specified -- which is the mistake the deleted
    `swebench_pro/` tree made when it string-matched patches instead of
    running anything.
    """
    d = swebench_pro_scripts_dir(instance_id)
    base = f"{SWEBENCH_PRO_SCRIPTS_RAW}/run_scripts/{instance_id}"
    run_script = _sbp_cached(f"{base}/run_script.sh",
                             os.path.join(d, "run_script.sh"), refresh=refresh)
    parser = _sbp_cached(f"{base}/parser.py",
                         os.path.join(d, "parser.py"), refresh=refresh)
    if "run_all_tests" not in run_script:
        raise RuntimeError(f"{instance_id}: fetched run_script.sh looks wrong "
                           f"(no `run_all_tests`); got {len(run_script)} chars")
    return {"run_script": run_script, "parser": parser,
            "env_exports": _sbp_env_exports(instance_id, refresh=refresh),
            "dir": d}


def sbp_restore_tests_cmd(problem):
    """The one command that materialises the graded test files.

    `before_repo_set_cmd` is four lines: reset, clean, checkout base, then
    `git checkout <solution_commit> -- <test files>`. Upstream's entry script
    takes **only the last line**, because the first three are exactly what the
    harness has already done, and re-running them after the candidate patch is
    applied would throw the patch away. That is the whole reason this is a
    function with a docstring instead of an inline `[-1]`.
    """
    raw = str(problem.get("before_repo_set_cmd") or "").strip()
    return raw.split("\n")[-1].strip() if raw else ""


def sbp_test_files(problem):
    """The test files the run script is pointed at, comma-joined as it expects."""
    return list(problem.get("selected_test_files_to_run") or [])


def sbp_required_tests(problem):
    """Every test name that must report PASSED for the row to count as resolved."""
    out, seen = [], set()
    for key in ("fail_to_pass", "pass_to_pass"):
        for name in (problem.get(key) or []):
            name = str(name)
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def load_swebench_pro_problems(split=SWEBENCH_PRO_DEFAULT_SPLIT, max_tasks=None,
                               apply_quarantine=True, languages=None):
    """Load SWE-bench Pro instances as a dictionary keyed by instance_id.

    `languages` filters on `repo_language`, whose values in the published split
    are "go" (280), "python" (266), "js" (165) and "ts" (20). It exists
    because the straitjacket harness's typed fact tier is profile-detected from
    test output: a Python row digests as `pytest/v1`, a mocha row as text. An
    evidence-gated arm reads that tier, so mixing languages inside one sweep
    mixes two qualities of routing signal under one arm name.

    Rows whose own gold patch cannot be scored in this environment are excluded
    using the file `tools/swebench_pro_preflight.py` writes. That file is
    environment-specific -- regenerate it per machine rather than copying it.
    """
    path = ensure_swebench_pro_dataset(split)
    excluded = load_swebench_pro_quarantine(split) if apply_quarantine else {}
    wanted = {str(x).lower() for x in (languages or [])}
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
            for key in _SBP_LIST_FIELDS:
                row[key] = _sbp_list(row.get(key))
            if wanted and str(row.get("repo_language") or "").lower() not in wanted:
                continue
            row["dataset_type"] = "swebench_pro"
            row["task_id"] = iid
            row["image_name"] = swebench_pro_image(row)
            # Upstream's images always clone into /app; the executor still
            # verifies it rather than trusting the convention.
            row["repo_workdir"] = "/app"
            problems[iid] = row
            if max_tasks and len(problems) >= max_tasks:
                break
    note = ""
    if skipped:
        note = (f" ({skipped} excluded by "
                f"{os.path.basename(swebench_pro_quarantine_path(split))}"
                " -- gold does not pass here)")
    elif apply_quarantine and not os.path.exists(swebench_pro_quarantine_path(split)):
        note = "  [no preflight run yet: python3 tools/swebench_pro_preflight.py --gold 5]"
    if wanted:
        note += f" [languages={','.join(sorted(wanted))}]"
    print(f"Loaded {len(problems)} SWE-bench Pro instances from {_rel(path)}{note}")
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
      - 'swebench-pro', 'swebench_pro', 'sbp'
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
    elif name in ("swebench_pro", "swe_bench_pro", "sbp", "swebenchpro"):
        s = split or SWEBENCH_PRO_DEFAULT_SPLIT
        return load_swebench_pro_problems(
            split=s, max_tasks=max_tasks,
            languages=[x for x in os.environ.get("SBP_LANGUAGES", "").split(",") if x])
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. Supported: "
                         "'bcb', 'webdev', 'classeval', 'featurebench', 'swebench-pro'.")
