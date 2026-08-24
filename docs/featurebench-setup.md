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

### 2.3 Disk

Images are **per repository, not per instance** — the fast split's 100 instances
draw on 18 images (lite 13, full 24). Repository images of this kind commonly run
10–30 GB in total; the upstream docs do not state a figure, so measure rather
than trust an estimate:

```bash
python3 tools/featurebench_preflight.py --pull
docker system df
```

Reclaim later with `docker image prune -a`.

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

**Step 1 — check what `repo_settings` actually holds.** The loader derives the
repository's in-image path from it, and a wrong binding would change what every
arm is scored on:

```bash
python3 tools/featurebench_preflight.py --settings
```

If the printed `repo_workdir` is not where the repo lives inside the image, bind
the correct key in `_fb_workdir` ([src/datasets.py](../src/datasets.py)). Gold
will fail on every row until it is right — which is exactly what step 3 catches.

**Step 2 — pull the images.**

```bash
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

FeatureBench problem statements run **6k–77k characters**. That is 10–100× a
BCB-Hard prompt, and input tokens dominate in a way they never did on the other
datasets. Modelled from [`src/config.py`](../src/config.py)'s pricing at a ~5k
token statement and ~8k token repair prompts:

| Arm | n=20 | n=100 |
|---|---|---|
| `fb_single_flash` | ~$2 | ~$10 |
| `fb_single_sonnet` | ~$3 | ~$13 |
| `fb_cascade` | ~$4 | ~$19 |
| `fb_evidence_gate` | ~$5 | ~$26 |
| `fb_plan_exec` | ~$3 | ~$16 |
| **`--group featurebench` total** | **~$17** | **~$84** |
| `fb_single_opus` (opt-in) | ~$7 | ~$33 |

Treat these as an order of magnitude. The N=148 routing study overshot its own
pre-run estimate by ~1.3×, and the statement-length spread here is much wider
than anything that estimate was built on. **Start at `--n 20`.**

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
