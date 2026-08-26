# Comparative TCO Report: `straitjacket` on `SWE-bench Pro` (N=20)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **SWE-bench Pro** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **S0b. Single: claude-sonnet-5 (3 rungs)** | `Claude Sonnet-5 x3` | Straitjacket contained digest ($0.00) | 6/20 (30.0%) | **6/20 (30.0%)** | `$3.3972` | `$0.0000` | **`$0.5662`** | `5782.7` |
| **S1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate, 3 rungs)** | `Gemini 3.7 Flash -> Claude Sonnet-5` | Straitjacket contained digest ($0.00) | 6/20 (30.0%) | **6/20 (30.0%)** | `$7.1195` | `$0.0000` | **`$1.1866`** | `12354.9` |
| **S2. Evidence gate: flash -> sonnet/opus (evidence gate, 3 rungs)** | `Gemini 3.7 Flash -> Claude Sonnet-5 / Claude Opus-5 (evidence gate)` | Straitjacket digest + evidence-gated escalation ($0.00) | 8/20 (40.0%) | **8/20 (40.0%)** | `$7.8168` | `$0.0000` | **`$0.9771`** | `13620.4` |

---

## 2. Attempt Diagnostics

What happened to every *attempt*, not every task. `Suite reached` is the share of attempts whose evidence came from the repository's own test run; the rest died at a guard before grading and say nothing about the model. `Avg partial` is the mean `test_pass_ratio` over graded attempts — the credit a binary resolved/not-resolved verdict discards. `Frontier` and `Degraded` are the router's own record: how many tasks actually reached the frontier rung, and how many were routed by a gate that wanted typed evidence and did not get it.

| Configuration | Attempts | Suite reached | Avg partial | Grounded | Frontier used | Degraded | Ungraded attempts, by cause |
|---|---|---|---|---|---|---|---|
| **S0b. Single: claude-sonnet-5 (3 rungs)** | 50 | 28/50 (56%) | `0.821` (n=16) | 20 | 0 | 0 | `apply_failed` × 22 |
| **S1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate, 3 rungs)** | 50 | 22/50 (44%) | `0.759` (n=16) | 20 | 15 | 0 | `apply_failed` × 25 · `no_patch` × 2 · `truncated_output` × 1 |
| **S2. Evidence gate: flash -> sonnet/opus (evidence gate, 3 rungs)** | 53 | 22/53 (42%) | `0.867` (n=18) | 20 | 16 | 1 | `apply_failed` × 27 · `truncated_output` × 2 · `no_patch` × 2 |

> [!WARNING]
> **Most attempts were never graded** — `S1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate, 3 rungs)`, `S2. Evidence gate: flash -> sonnet/opus (evidence gate, 3 rungs)` reached the repository's test suite on fewer than half of their attempts. Their pass rates measure whether a candidate patch could be *applied*, not whether it resolved the issue. Read the guard-failure column before reading the pass rate.

---

## 3. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **S0b. Single: claude-sonnet-5 (3 rungs)** | straitjacket | `pytest/v1, pytest/v2, text/v1` | 28 | `216,114` | **`17,554`** | `12,919` | `-4,635` (-36%) |
| **S1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate, 3 rungs)** | straitjacket | `pytest/v1, pytest/v2, text/v1` | 22 | `204,923` | **`17,565`** | `14,287` | `-3,278` (-23%) |
| **S2. Evidence gate: flash -> sonnet/opus (evidence gate, 3 rungs)** | straitjacket | `pytest/v1, pytest/v2, text/v1` | 22 | `167,037` | **`16,755`** | `15,706` | `-1,049` (-7%) |

---

## 4. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `S0b. Single: claude-sonnet-5 (3 rungs)` is the cheapest per solved task at `$0.5662`; `S2. Evidence gate: flash -> sonnet/opus (evidence gate, 3 rungs)` has the highest pass rate at 40%.
