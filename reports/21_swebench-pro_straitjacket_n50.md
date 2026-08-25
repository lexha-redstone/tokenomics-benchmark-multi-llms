# Comparative TCO Report: `straitjacket` on `SWE-bench Pro` (N=49)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **SWE-bench Pro** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **S0a. Single: gemini-3.7-flash low (2 rungs)** | `Gemini 3.7 Flash (low) x2` | Straitjacket contained digest ($0.00) | 1/49 (2.0%) | **1/49 (2.0%)** | `$5.1890` | `$0.0000` | **`$5.1890`** | `12886.6` |
| **S0b. Single: claude-sonnet-5 (2 rungs)** | `Claude Sonnet-5 x2` | Straitjacket contained digest ($0.00) | 0/50 (0.0%) | **0/50 (0.0%)** | `$4.1643` | `$0.0000` | **`$0.0000`** | `6549.6` |
| **S1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate, 2 rungs)** | `Gemini 3.7 Flash -> Claude Sonnet-5` | Straitjacket contained digest ($0.00) | 1/47 (2.1%) | **1/47 (2.1%)** | `$6.3147` | `$0.0000` | **`$6.3147`** | `9569.1` |
| **S2. Evidence gate: flash -> sonnet/opus (evidence gate, 2 rungs)** | `Gemini 3.7 Flash -> Claude Sonnet-5 / Claude Opus-5 (evidence gate)` | Straitjacket digest + evidence-gated escalation ($0.00) | 1/40 (2.5%) | **1/40 (2.5%)** | `$2.6101` | `$0.0000` | **`$2.6101`** | `6890.4` |
| **S3. H2: opus-5 plans first, 3.7-flash implements and repairs (2 rungs)** | `Claude Opus-5 plan + Gemini 3.7 Flash exec x2` | Straitjacket contained digest ($0.00) | 0/50 (0.0%) | **0/50 (0.0%)** | `$4.9070` | `$0.0000` | **`$0.0000`** | `7053.5` |

---

## 2. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **S0a. Single: gemini-3.7-flash low (2 rungs)** | straitjacket | `pytest/v1, pytest/v2, text/v1` | 10 | `15,262` | **`4,234`** | `4,015` | `-219` (-5%) |
| **S0b. Single: claude-sonnet-5 (2 rungs)** | straitjacket | `pytest/v2` | 5 | `32,517` | **`4,421`** | `4,380` | `-41` (-1%) |
| **S1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate, 2 rungs)** | straitjacket | `pytest/v1, pytest/v2, text/v1` | 15 | `78,714` | **`4,017`** | `4,414` | `+397` (+9%) |
| **S2. Evidence gate: flash -> sonnet/opus (evidence gate, 2 rungs)** | straitjacket | `pytest/v1, pytest/v2` | 6 | `14,163` | **`2,304`** | `2,896` | `+592` (+20%) |
| **S3. H2: opus-5 plans first, 3.7-flash implements and repairs (2 rungs)** | straitjacket | `pytest/v2, text/v1` | 8 | `9,078` | **`3,226`** | `3,223` | `-3` (-0%) |

---

## 3. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `S2. Evidence gate: flash -> sonnet/opus (evidence gate, 2 rungs)` is the cheapest per solved task at `$2.6101`; `S2. Evidence gate: flash -> sonnet/opus (evidence gate, 2 rungs)` has the highest pass rate at 2%.
