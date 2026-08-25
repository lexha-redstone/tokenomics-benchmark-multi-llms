# FeatureBench on Linux — setup and first run

> **What this dataset is for.** It is the only benchmark in this repository with
> an *expensive* oracle. BCB-Hard and ClassEval run their tests in a sandbox for
> $0 in milliseconds; a FeatureBench attempt applies a diff inside the
> repository's own container and runs pytest, which upstream measures at 57.2 s
> on gold patches. That is the P4 property
> [§7 of the dataset survey](pattern-dataset-selection.md) says is missing
> everywhere else, and it is what makes **H2** testable:
>
> > **H2.** As the oracle gets more expensive or more partial, the cascade's
> > advantage shrinks, because *fail → escalate* stops being a free routing
> > signal.
>
> Every "escalate rather than plan" finding in this repository was measured
> where failing is free. This is where that stops being true.

---

## 1. What is *not* required

Worth stating first, because the obvious assumption is wrong and costs an hour.

**The upstream `featurebench` package and its `fb` CLI are not needed.** That
tooling is built around a two-stage flow — `fb infer` drives an agent framework
inside the container, then `fb eval` grades the emitted `patch.diff`. This
repository's arms are not an agent framework; they are a fixed repair ladder
whose routing signal *is* the test result, so the oracle has to sit inside the
loop rather than after it. `src/evaluator.py` therefore drives Docker directly:
one container per task, reused across that task's attempts.

So there is no `uv sync`, no second toolchain, and no `pip install featurebench`.
What you need is Docker and the repo's existing environment.

---

## 2. Prerequisites

### 2.1 Docker Engine

On Linux use Docker Engine, not Desktop. Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
```

RHEL/Fedora/Amazon Linux:

```bash
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io
sudo systemctl enable --now docker
```

### 2.2 Run Docker without sudo

The benchmark shells out to `docker` as your user. Without this every call fails
on the socket permission:

```bash
sudo usermod -aG docker "$USER"
newgrp docker          # or log out and back in
docker info            # must succeed with no sudo
```

### 2.3 Disk — measure before you pull

**Ask the registry first.** This needs no Docker and no download:

```bash
python3 tools/featurebench_preflight.py --disk
```

It counts the distinct `image_name` values in the split and queries Docker Hub
for each one's compressed size, then prints a projected on-disk range.

Two corrections to what the upstream README implies, both checked against the
registry:

- **The images are per *instance*, not per repository.** `fb pull --mode fast`
  documents 18 images, but the dataset's own `image_name` field looks like
  `libercoders/featurebench-specs_packaging-instance_c393a6a8` — and
  `libercoders/featurebench-specs_packaging` (no instance suffix) returns 404,
  so the suffix is part of the repository name. Do not assume 18; run `--disk`.
- **They are large.** The one instance image measured directly,
  `featurebench-specs_packaging-instance_c393a6a8`, reports a `full_size` of
  **10.18 GB compressed** — for `pypa/packaging`, one of the smallest libraries
  in the set. These images evidently carry a heavy common base.

So plan for **hundreds of GB, not tens**, and read the `--disk` output before
committing. Two effects move the real number in opposite directions: images
decompress on disk (up ~1.4–2.5×), while layers shared between images are stored
**once** (down, potentially by a lot, since these are built from a common base).

`docker system df` after the pull is the only exact answer:

```bash
python3 tools/featurebench_preflight.py --pull
docker system df
```

If the projection is more disk than you have, pull nothing up front and run a
small `--n` instead: images are fetched on first use, so a 20-instance sweep
only ever materialises the images those 20 rows need. Reclaim with
`docker image prune -a`.

**`--pull` is safe to interrupt.** Two mechanisms make a re-run continue rather
than restart:

- Images already in the local store are **skipped outright** — the tool checks
  with `docker image inspect` before asking the registry for anything.
- `docker pull` keeps whatever layers it finished. Resumption is at **layer**
  granularity, not byte: a layer that was mid-flight when you stopped starts
  over, completed layers do not. With ~10 GB images that can still mean
  re-fetching a few hundred MB, but never the whole image.

Check what you already have at any point:

```bash
docker images | grep featurebench
docker system df
```

Docker's own progress output is left on the terminal deliberately. An earlier
version captured it, which left the screen silent for minutes at a time and made
a healthy multi-GB pull look hung. The per-image cap is `--pull-timeout`
(default 7200 s) for the same reason — a short cap kills a working download.

### 2.4 Architecture

The published images target x86-64. On arm64 you would be running everything
under emulation, which multiplies the 57.2 s/instance figure. Use an x86-64
Linux host.

### 2.5 The repo's own environment

Nothing new, but both of these must hold or the arms refuse to run:

```bash
pip install -r requirements.txt      # includes ctx-harness
python3 -m src.straitjacket          # must report backend=library
```

`fb_evidence_gate` reads the harness's **typed** fact tier. Under any backend but
`library` there is nothing to gate on: the arm silently degrades into
`fb_cascade`, and it sets `routing.degraded = true` so that shows up in the
results instead of being published as a finding.

---

## 3. The preflight — run this before spending anything

Same discipline as ClassEval, for the same reason. ClassEval taught it cheaply:
8 of 100 classes cannot be scored on a provisioned machine and 12 more on a bare
one, and finding that out mid-sweep means two machines measured different task
sets while reporting comparable-looking pass rates. FeatureBench has more ways to
fail, not fewer — an image that will not pull, a `test_patch` that will not
apply, a workdir that is not where the loader guessed.

**Step 1 — see what `repo_settings` actually holds.** Runs without Docker:

```bash
python3 tools/featurebench_preflight.py --settings
```

Checked against the real fast split, the 40 keys are `repository`, `commit`,
`base_image`, `install`, `pip_packages`, **`test_cmd`**, `timeout_run`,
`library_name` … and **nothing path-like**. So:

- The **test command comes from the row** (`test_cmd`), not from a hardcoded
  pytest line — otherwise the arms would be scored on a command the benchmark
  never specified. Timeouts come from `timeout_run` the same way.
- The **repository's in-image path is discovered at run time**, not guessed:
  `FeatureBenchEnv` reads the image's own `WORKDIR` via `docker inspect`, then
  confirms it with `git rev-parse --show-toplevel`. `/workspace/<library_name>`
  is only a last resort if both fail, and step 3 catches it if it is wrong.

**Step 2 — size the download, then pull.** See §2.3 first: these images are far
larger than the upstream README implies.

```bash
python3 tools/featurebench_preflight.py --disk     # no Docker needed
python3 tools/featurebench_preflight.py --pull
```

**Step 3 — run FeatureBench's own gold patches.**

```bash
python3 tools/featurebench_preflight.py --n 5          # look first
python3 tools/featurebench_preflight.py --write        # then record
```

It applies each row's reference `patch`, runs the row's tests, and writes every
row gold cannot resolve to `featurebench/data/quarantine-<split>.json` with a
typed reason — `missing_image` names a pull failure, `gold_patch_conflict` a
diff that would not apply, and so on. The loader honours that file. Rows are
excluded, never edited. It exits **2** while rows are unscorable and unrecorded,
so a sweep cannot start on top of an unknown environment.

**The quarantine file is environment-specific. Regenerate it on each machine; do
not copy it between them.**

Level 2 rows ship no reference patch, so there is nothing to verify and nothing
to blame on the environment — they are reported as `SKIP` and left in.

---

## 4. Running the sweep

```bash
# Smoke: two rows, the recommended shape and the shape it has to beat. ~$1.
python3 run_benchmark.py --dataset featurebench --n 2 \
    --variants fb_evidence_gate,fb_cascade --no-cache

# The comparison. Five arms.
python3 run_benchmark.py --dataset featurebench --group featurebench \
    --n 20 --report --no-cache
```

| Arm | What it is | Why it is here |
|---|---|---|
| `fb_single_flash` | `gemini-3.7-flash` (low) × 3 | the cheap single |
| `fb_single_sonnet` | `claude-sonnet-5` × 3 | best `$/solved` of any arm at BCB-Hard N=148 |
| `fb_cascade` | flash → sonnet → opus, escalate when a rung fails | the `r6` shape — tied the plain frontier baseline exactly at N=148 |
| `fb_evidence_gate` | same tiers, escalate when the digest reads `broad`/`stalled` | **the recommended shape** — `r9`, 96% of frontier accuracy for 74% of frontier spend at N=148 |
| `fb_plan_exec` | **opus-5 plans first**, flash implements and repairs | **the H2 challenger** |
| `fb_single_opus` | `claude-opus-5` × 3 | the ceiling. **Not** in `--group featurebench`; opt-in, see below |

Two design choices are deliberate and worth knowing before reading a result:

- **`gemini-3.5-flash-lite` is not a rung.** On BCB-Hard it was a sensible first
  attempt because a wasted attempt cost a fraction of a cent. Here a wasted
  attempt costs a container run, and a multi-file feature is far outside what
  Lite managed on single functions. Spending the scarce resource on a rung that
  cannot plausibly succeed is the mistake H2 is about.
- **No `medium`/`high` thinking ladder.** N=148 measured `gemini-3.7-flash` at
  `medium` costing 33% more than `claude-opus-5` while solving 14 fewer tasks.
  Escalate the model, not the budget.

**Every arm makes exactly 3 oracle calls.** That is the scarce resource here, so
it is the one held constant. `fb_plan_exec` buys one extra *LLM* call for its
plan, which shows up in dollars where it belongs — that is the H2 contrast: the
same frontier model, spent before the first oracle call instead of after the
last.

Two consequences of holding that constant, both deliberate and both enforced by
[`tests/test_featurebench.py`](../tests/test_featurebench.py):

- **Escalation is a one-way ratchet.** No arm ever drops to a cheaper rung.
  N=100 measured a de-escalating repair turn rescuing 16% of failures against
  41% for an escalating one (z = +3.55, p = 0.0004), so building the losing
  direction into an arm would be designing in a known defect.
- **A spare attempt is spent on the rung already held.** `fb_evidence_gate` can
  jump straight to the frontier on its first repair and then have budget left
  with nothing above it; it re-runs the frontier rather than handing the attempt
  back, because an arm that quietly returns a container run reads as cheaper for
  a reason unrelated to its routing policy.

The frontier single is opt-in for the same reason as ClassEval's:

```bash
python3 run_benchmark.py --dataset featurebench --variants fb_single_opus --n 20
```

---

## 5. Cost, and how to not be surprised by it

FeatureBench problem statements run **6k–77k characters** — the sample row is
43,308, i.e. ~11k tokens. That is 10–100× a BCB-Hard prompt, and input tokens
dominate in a way they never did on the other datasets. Modelled from
[`src/config.py`](../src/config.py)'s pricing at ~10k-token statements and
~15k-token repair prompts:

| Arm | n=20 | n=100 |
|---|---|---|
| `fb_single_flash` | ~$3 | ~$17 |
| `fb_single_sonnet` | ~$5 | ~$23 |
| `fb_cascade` | ~$7 | ~$33 |
| `fb_evidence_gate` | ~$9 | ~$45 |
| `fb_plan_exec` | ~$5 | ~$27 |
| **`--group featurebench` total** | **~$29** | **~$146** |
| `fb_single_opus` (opt-in) | ~$12 | ~$57 |

Treat these as an order of magnitude. The N=148 routing study overshot its own
pre-run estimate by ~1.3×, and the statement-length spread here (6k–77k chars)
is far wider than anything that estimate was built on. **Start at `--n 20`.**

`fb_evidence_gate` is the most expensive multi-model arm rather than the
cheapest, which inverts its N=148 position. That is expected and is not a bug:
when the gate fires early it holds the frontier rung for the remaining
attempts, and here there is no expensive middle rung for it to skip — on
BCB-Hard its saving came precisely from skipping `gemini-3.7-flash (medium)`.
Whether it earns that back in resolved tasks is the measurement.

Wall-clock is the other budget: 3 oracle calls × ~57 s × tasks × arms. At n=20
across five arms that is roughly 5 container-hours before model latency.

---

## 6. Reading the result

The same three numbers as the routing study, plus one FeatureBench adds:

- **pass rate** — `resolved`: pytest exits 0 over the fail-to-pass and
  pass-to-pass files together. This is FeatureBench's own Resolved Rate.
- **`$/solved`** — against `fb_single_opus`, the same-budget frontier baseline.
- **frontier yield** — of the tasks handed to opus, how many it solved.
- **`test_pass_ratio`** — the fraction of executed test cases that passed, read
  off pytest's summary. Upstream reports a *Passed Rate* beside Resolved Rate
  for a good reason: on a dataset where frontier models resolve 20–47%, a binary
  verdict makes every cheap arm read as an undifferentiated zero. It is named
  `test_pass_ratio` rather than `passed_rate` because the upstream denominator
  (fail-to-pass tests only) is not something this code can verify from pytest
  output alone — so do not present it as FeatureBench's published metric.

**What would settle H2:** the first-repair rescue rate split by whether the
repair turn escalated, exactly as
[`tools/analyze_patterns.py`](../tools/analyze_patterns.py) computes it today.

- If escalation still dominates at 57 s and a container per attempt, the finding
  generalises past its cheap-oracle origins and the README's recommendation
  stands as written.
- If `fb_plan_exec` closes the gap or overtakes, H2 holds, the current guidance
  is scoped to cheap-oracle workloads, and the routing matrix needs a column for
  retry cost.

Either outcome is worth the run — the same standard ClassEval was held to.

### A cross-check worth doing once

This repository scores rows with its own container runner rather than `fb eval`,
because the ladder needs the oracle inside the loop. That is a fidelity risk, and
the cheap way to close it is to grade one sweep both ways: install the upstream
package in a throwaway environment, feed it the final patches, and confirm the
resolved set matches. Check the expected prediction format with `fb eval --help`
before writing the file — it is not documented here because it has not been
verified against a live install.

---

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| `docker: permission denied` | §2.2 — you are not in the `docker` group |
| every row quarantines as `missing_image` | images not pulled, or no registry access; run `--pull` and read its output |
| every row quarantines as `gold_patch_conflict` | `repo_workdir` is wrong — run `--settings` and rebind `_fb_workdir` |
| `patch did not apply` from a model, repeatedly | expected and informative; it is fed back as evidence, and a model that cannot emit an applicable diff is a real finding on this dataset |
| `fb_evidence_gate` results look identical to `fb_cascade` | check `routing.degraded` — the backend is probably not `library` |
| very slow | confirm x86-64 (§2.4), and that images are pre-pulled so the first task is not also a download |
