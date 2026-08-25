# SWE-bench Pro — running it from Python, on Docker

> **The short answer to "is this possible?"** Yes, and with less local
> machinery than FeatureBench needs. Upstream ships everything that decides a
> verdict — the image, the git command that restores the graded tests, the
> script that runs each repository's suite, and the parser that reads its
> output. `src/evaluator.py` drives Docker directly and calls those four
> things; nothing about the grading is reimplemented here. Assuming a working
> Docker daemon, the whole path is Python plus `docker exec`.

---

## 1. Why this dataset replaced FeatureBench for the H2 study

FeatureBench was adopted to test **H2** — *as the oracle gets more expensive,
does the cascade's advantage shrink, so that front-loaded planning finally
pays?* The arms are fine. The rows are the problem: a FeatureBench row is only
scorable when the repository's own `test_patch` applies to the image it ships
with, and the harness has to rebuild the graded tree itself before anything
runs. When that fails it fails for *every* arm, which is not a hard task, it is
a missing measurement.

SWE-bench Pro is the same shape of task — multi-file work in a real repository,
graded by that repository's own tests inside a per-instance Docker image — with
that step removed rather than worked around. It is also a *harder* oracle,
which is what H2 is about: an attempt runs a real suite (`npm install` and
all), and frontier agents resolve roughly 20–40% of the split.

| | FeatureBench | SWE-bench Pro |
|---|---|---|
| Graded tests come from | `test_patch`, applied locally, must not conflict | `git checkout <solution> -- <files>`, shipped in the row |
| Test command | row's `test_cmd`, then pytest | upstream's per-instance `run_script.sh` |
| Verdict | pytest exit code | upstream's per-instance `parser.py`, then a set comparison |
| Rows | 100 (fast split) | 731 (public split) |
| Languages | Python | go 280 · python 266 · js 165 · ts 20 |

---

## 2. What is *not* required

* **Modal.** Upstream's default runtime is a Modal sandbox and its
  `--use_local_docker` path is a beta flag. This repository mirrors the local
  Docker path only: the routing question here is about policy and dollars, and
  a second scheduler in the loop is one more thing that can fail in a way the
  numbers quietly absorb.
* **A clone of `scaleapi/SWE-bench_Pro-os`.** The per-instance `run_script.sh`,
  `parser.py` and Dockerfiles are fetched on demand and cached under
  `swebench_pro/run_scripts/<instance_id>/` (a few kB each).
* **The `datasets` library.** The split is pulled through the HuggingFace rows
  API into `swebench_pro/data/SWE-bench_Pro-test.jsonl` (~24 MB, 731 rows).
* **`pip install swebench`.** Nothing from the SWE-bench toolchain is imported.

---

## 3. Prerequisites

### 3.1 Docker

A running daemon the current user can reach without `sudo`. Check with the
preflight rather than by hand — it prints the server version and stops early:

```bash
python3 tools/swebench_pro_preflight.py --ready --n 5
```

### 3.2 Architecture

Upstream publishes **linux/amd64 images only**. On Apple Silicon the executor
passes `--platform linux/amd64` automatically (`_sbp_default_platform`), which
means every container runs under emulation and a suite that takes 3 minutes on
x86 can take 15. Override with `SBP_PLATFORM` if you know better. For a real
sweep, use an x86-64 machine.

### 3.3 Disk

One image per instance, and they are not small. Measured over a random sample
of 14 rows (compressed size as Docker Hub reports it; on-disk is larger):

| | compressed |
|---|---|
| smallest (ansible, qutebrowser) | 0.5 GB |
| median | 1.1 GB |
| largest (protonmail/webclients) | 4.2 GB |

Pulling all 731 is not a sensible first move. Pull a slice, and pass the ids
that are ready to the runner:

```bash
python3 tools/swebench_pro_preflight.py --languages python --n 20 --pull
python3 tools/swebench_pro_preflight.py --ready --ready-out /tmp/ready.txt
```

### 3.4 Network stays on

Several run scripts install dependencies at test time — NodeBB's runs
`npm install` before every suite. `--network none` does not harden the run,
it fails it. The executor defaults to `bridge`; set `SBP_NETWORK=none` only
for a slice you have already proved is self-contained.

---

## 4. The preflight — run this before spending anything

A gold run is the only honest smoke test of the harness. It exercises every
step an arm's attempt takes — container start, reset, `git apply`, the
graded-test restore, upstream's run script, upstream's parser, the resolution
rule — against a patch that is known to be correct. **If gold does not pass,
the harness is wrong, not the model.**

```bash
python3 tools/swebench_pro_preflight.py --languages python --n 3 --pull
python3 tools/swebench_pro_preflight.py --languages python --gold 3
```

The executor captures every test run through the straitjacket harness, so a
gold run needs it too (`pip install ctx-harness`). The preflight checks that
before it spends anything and says so, rather than reporting one "execution
error" per row.

Rows whose own reference patch does not resolve here are written to
`swebench_pro/data/quarantine-test.json` with a reason, and the loader honours
it:

```bash
python3 tools/swebench_pro_preflight.py --gold 0 --write   # whole split
```

**That file is environment-specific. Regenerate it on each machine; do not copy
it between them.** A `missing_image` entry means `docker pull` failed, not that
the task is broken.

---

## 5. What one attempt actually does

Per candidate patch, inside a container that is reused across the task's three
attempts:

1. `git reset --hard <base_commit>`, `git checkout -f`, `git clean -fdq`.
2. Write the diff and apply it, strictest strategy first: `git apply`, then
   `--recount` (recomputes miscounted `@@` headers — the commonest defect in a
   model-authored diff and the one that says least about the edit), then
   `--ignore-whitespace -C1`, then a `--3way` merge against the blobs the
   repository already has, then `patch --batch --forward --fuzz=5 -p1`. The
   worktree is reset between failed strategies, because `--3way` writes
   conflict markers *before* it exits non-zero. None of this relaxes grading:
   the candidate still has to make the suite pass. A patch that fails every
   strategy is fed back with the full `git apply --verbose` log — the failing
   file, the hunk, the context block it searched for — not as a fixed sentence.
3. Run **the last line of `before_repo_set_cmd`** — `git checkout <solution> --
   <test files>`. This is the anti-cheat, and the order matters: the graded
   tests land *over* whatever the candidate did to them. Its failure is an
   error, never a silent grade against the repository's original tests.
4. `rm -f /workspace/output.json`, then `bash run_script.sh <files>` with
   stdout/stderr to files, then `parser.py`. The delete is load-bearing: the
   container is reused, so a stale verdict file is the *previous rung's*
   result.
5. The logs are echoed so the straitjacket harness captures exactly the bytes
   the parser read. That capture is the digest the repair turn receives.
6. Resolved ⟺ every name in `fail_to_pass ∪ pass_to_pass` came back `PASSED` —
   upstream's `(f2p | p2p) <= passed`, so breaking a `pass_to_pass` test fails
   the row even when the issue was fixed. `test_pass_ratio` carries the partial
   credit that a binary verdict throws away.

---

## 6. Running the sweep

```bash
python3 run_benchmark.py --dataset swebench-pro --group sbp --n 20 --report
```

Only some images pulled? Name the rows explicitly:

```bash
python3 run_benchmark.py --dataset swebench-pro --group sbp --tasks @/tmp/ready.txt --report
```

The arms are the FeatureBench five, unchanged in shape so the two datasets'
rows mean the same thing:

| Arm | What it is |
|---|---|
| `sbp_single_flash` / `sbp_single_sonnet` | one model, three attempts |
| `sbp_cascade` | flash → sonnet → opus, escalate on every failure |
| `sbp_evidence_gate` | same tiers, escalate when the digest says the failure is hard — the recommended shape from N=148 |
| `sbp_plan_exec` | **the H2 challenger**: opus plans before any test runs, flash implements and repairs |
| `sbp_single_opus` | frontier baseline; opt-in via `--variants sbp_single_opus` |

Every arm makes exactly **three oracle calls** — three container test runs.
That is the resource H2 says is scarce, so it is the one held constant;
`sbp_plan_exec` buys one extra *LLM* call, which shows up in dollars.

---

## 6b. Repository grounding -- why the prompt quotes source

Upstream's prompt is a statement, a requirements block and an interface block,
and `src/swebench_pro.py` reproduces all three verbatim. What upstream *also*
gives its agents, and what a one-shot prompt does not, is the repository.
Measured over the published 731-row split:

| | |
|---|---|
| reference-patch files named anywhere in the three blocks | 19.8% |
| rows where **every** changed file is named | **8.8%** |
| median reference patch | 9 hunks across 4 files |
| patches that only create new files (no context needed) | 1.8% |

`git apply` needs every context line of every hunk to match a file the model
was never shown, so a blind prompt has a localisation ceiling near 9% before a
single hunk is written. The published 20-40% resolve rates are *agent* numbers,
measured with file access; they are not comparable to a blind one-shot arm.

The container is already running and already holds the tree at `base_commit`,
so `_ladder` quotes the likely files into the prompt before spending a token.
Paths come from the row's own `Path:` lines, its graded test files and any
path-shaped token in the prose; when that locates fewer than three files, the
salient identifiers are `git grep`-ed for. Source is read with
`git show <base_commit>:<path>` -- never from the worktree, which holds whatever
the previous attempt applied.

Set `SBP_GROUNDING_CHARS=0` to reproduce the blind prompt byte for byte. That
is the A/B leg, and `tests/test_swebench_pro_grounding.py` pins it.

---

## 6c. Reading the diagnostics before reading the pass rate

A containerised dataset has two failure owners, and a pass rate cannot tell
them apart. Every report now carries an **Attempt Diagnostics** section:

| Column | What it answers |
|---|---|
| `Suite reached` | share of *attempts* whose evidence came from the repository's own test run. The rest died at a guard and say nothing about the model |
| `Avg partial` | mean `test_pass_ratio` over graded attempts, with the count it was averaged over |
| `Frontier used` | how many tasks actually reached the frontier rung |
| `Degraded` | tasks routed by a gate that wanted typed evidence and did not get it |
| `Dominant guard failure` | which of `apply_failed`, `no_patch`, `not_a_diff`, `no_hunk`, `truncated_output`, `container_unavailable`, `restore_failed`, `execution_error` accounted for most attempts |

Two warnings are emitted automatically and are not advisory:

* **Most attempts were never graded** -- under 50% suite reach. The pass rate
  measures whether a patch could be *applied*, not whether it resolved
  anything.
* **Frontier rung never invoked** -- an arm names a frontier model in its model
  column and never called it. It did not test the architecture it is named
  after.

A third warning fires when arms in one report were scored on different task
sets, which happens whenever a dispatch failure drops a task from one arm and
not the others. Compare on the intersection it names.

---

## 7. One caveat that changes what a number means

The straitjacket digest's typed fact tier is **profile-detected from the test
output**, and this split spans four languages. A Python row's pytest output
digests as a typed profile the evidence gate can read; a mocha row's JSON
reporter blob does not, and `routing.degraded` is set on that row.

So a mixed-language `sbp_evidence_gate` sweep is two arms wearing one name. Run
it per language and say which in the report:

```bash
python3 tools/swebench_pro_preflight.py --languages python --gold 0 --write
SBP_LANGUAGES=python python3 run_benchmark.py --dataset swebench-pro --group sbp --n 30 --report
```

---

## 8. Environment variables

| Variable | Default | What it does |
|---|---|---|
| `SBP_LANGUAGES` | *(all)* | comma-separated `repo_language` filter for `run_benchmark.py` |
| `SBP_TIMEOUT` | `1800` | outer seconds for one attempt (run scripts carry their own inner timeouts) |
| `SBP_NETWORK` | `bridge` | container network; `none` breaks any suite that installs at test time |
| `SBP_PLATFORM` | auto | `linux/amd64` on arm64 hosts, empty elsewhere |
| `SBP_DOCKERHUB_USER` | `jefzda` | account hosting `sweap-images` |
| `SBP_INTEGRATION` | unset | `1` runs the Docker-backed test in `tests/test_swebench_pro.py` |
| `SBP_MAX_ORACLE_CALLS` | `3` | container test runs per task. Must exceed the rung count or the frontier tier is unreachable -- the registry refuses to load otherwise |
| `SBP_GROUNDING_CHARS` | `60000` | characters of repository source quoted into the solver prompt. `0` restores the blind prompt |
| `SBP_GROUNDING_FILE_CHARS` | `16000` | per-file cap, so one vendored bundle cannot eat the budget |
| `SBP_GROUNDING_MAX_FILES` | `12` | hard ceiling on how many files are quoted |
| `DISPATCH_MAX_ATTEMPTS` | `3` | retries per model call. At `1` a single 503 drops the task from that arm's denominator only |
| `DISPATCH_TIMEOUT_FLOOR` / `_PER_1K` / `_CAP` | `120` / `20` / `900` | request deadline, scaled with the output budget |

---

## 9. Troubleshooting

**`docker run` starts and immediately exits.** The images set
`ENTRYPOINT ["/bin/bash"]`, so a bare `docker run <image> sleep infinity` is
read as `bash sleep infinity` — bash looks for a *script called* `sleep`. The
executor passes `--entrypoint /bin/bash … -c "sleep infinity"`; if you are
reproducing by hand, do the same.

**`no matching manifest for linux/arm64`.** Apple Silicon without the platform
override. Set `SBP_PLATFORM=linux/amd64`.

**Gold fails with an empty test list (`reported: 0`).** The suite died before
reporting anything — usually a dependency install that needed network
(`SBP_NETWORK`) or an inner timeout. Look at the evidence string in the
quarantine entry.

**Gold fails with `test_pass_ratio` just under 1.0.** A `pass_to_pass` test the
image cannot satisfy on this machine. That is a real environment defect, and a
quarantine entry is the right outcome — not a loosened grading rule.

**`could not restore the graded test files`.** The row's checkout command names
a path the image's clone does not have. Quarantine it; grading it anyway would
run the repository's own tests and measure nothing.

---

## 10. Tests

```bash
python3 -m pytest tests/test_swebench_pro.py -q          # no Docker, no network
SBP_INTEGRATION=1 python3 -m pytest tests/test_swebench_pro.py -q -k gold
```

The offline set pins the things that would produce a *wrong number* rather than
an error: the list columns are Python literals and not JSON, only the last line
of `before_repo_set_cmd` is used, the graded tests are restored after the patch,
a stale `output.json` is deleted, a regression in `pass_to_pass` fails the row,
and every arm spends exactly three oracle calls. One test executes the
generated attempt script in a real shell against stub scripts; another runs
upstream's own parser over canned mocha output and checks the names it invents
are the names the dataset requires — the assumption everything else rests on.
