# Comparative TCO Report: `straitjacket` on `BigCodeBench-Hard` (N=1)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **BigCodeBench-Hard** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **R1. 3.7-Flash solo (thinking=low)** | `Gemini 3.7 Flash (low) x3` | Straitjacket contained digest ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0129` | `$0.0000` | **`$0.0129`** | `1328.0` |
| **R2. 3.7-Flash solo (thinking=medium)** | `Gemini 3.7 Flash (medium) x3` | Straitjacket contained digest ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0207` | `$0.0000` | **`$0.0207`** | `2665.0` |
| **R3. 3.7-Flash solo (thinking=high)** | `Gemini 3.7 Flash (high) x3` | Straitjacket contained digest ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0716` | `$0.0000` | **`$0.0716`** | `9444.0` |
| **R4. Gemini ladder: Lite -> 3.7(low) -> 3.7(medium)** | `Lite -> 3.7 Flash low -> 3.7 Flash medium` | Straitjacket contained digest ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0024` | `$0.0000` | **`$0.0024`** | `894.0` |
| **R5. Gemini thinking ladder: 3.7 low -> medium -> high (no Lite)** | `3.7 Flash low -> medium -> high` | Straitjacket contained digest ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0223` | `$0.0000` | **`$0.0223`** | `2588.0` |
| **R6. Gemini ladder -> Opus-5 (only after every Gemini rung fails)** | `Lite -> 3.7 low -> 3.7 medium -> Opus-5` | Straitjacket contained digest ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0024` | `$0.0000` | **`$0.0024`** | `886.0` |
| **R7. 3.7(medium) -> Opus-5 on the first failure (aggressive)** | `3.7 Flash medium -> Opus-5` | Straitjacket contained digest ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0317` | `$0.0000` | **`$0.0317`** | `4131.0` |
| **R8. 3.7 low -> medium -> Opus-5 after two failures** | `3.7 low -> 3.7 medium -> Opus-5` | Straitjacket contained digest ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0266` | `$0.0000` | **`$0.0266`** | `3156.0` |
| **R9. Gemini ladder -> Opus-5 when the digest says the failure is hard** | `Lite -> 3.7 low -> 3.7 medium -> Opus-5 (evidence gate)` | Straitjacket digest + evidence-gated escalation ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0025` | `$0.0000` | **`$0.0025`** | `943.0` |
| **R10. Gemini ladder -> Opus-5 re-solves from scratch (not a repair)** | `Lite -> 3.7 low -> 3.7 medium -> Opus-5 (fresh solve)` | Straitjacket contained digest ($0.00) | 1/1 (100.0%) | **1/1 (100.0%)** | `$0.0026` | `$0.0000` | **`$0.0026`** | `976.0` |

---

## 2. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **R1. 3.7-Flash solo (thinking=low)** | straitjacket | `unittest/v1` | 2 | `792` | **`181`** | `528` | `+347` (+66%) |
| **R2. 3.7-Flash solo (thinking=medium)** | **UNRECORDED** | `unittest/v1` | 1 | `264` | **`0`** | `0` | `+0` |
| **R3. 3.7-Flash solo (thinking=high)** | **UNRECORDED** | `unittest/v1` | 1 | `264` | **`0`** | `0` | `+0` |
| **R4. Gemini ladder: Lite -> 3.7(low) -> 3.7(medium)** | **UNRECORDED** | `unittest/v1` | 1 | `24` | **`0`** | `0` | `+0` |
| **R5. Gemini thinking ladder: 3.7 low -> medium -> high (no Lite)** | straitjacket | `unittest/v1` | 2 | `792` | **`181`** | `528` | `+347` (+66%) |
| **R6. Gemini ladder -> Opus-5 (only after every Gemini rung fails)** | **UNRECORDED** | `unittest/v1` | 1 | `24` | **`0`** | `0` | `+0` |
| **R7. 3.7(medium) -> Opus-5 on the first failure (aggressive)** | **UNRECORDED** | `unittest/v1` | 1 | `254` | **`0`** | `0` | `+0` |
| **R8. 3.7 low -> medium -> Opus-5 after two failures** | straitjacket | `unittest/v1` | 2 | `792` | **`181`** | `528` | `+347` (+66%) |
| **R9. Gemini ladder -> Opus-5 when the digest says the failure is hard** | **UNRECORDED** | `unittest/v1` | 1 | `24` | **`0`** | `0` | `+0` |
| **R10. Gemini ladder -> Opus-5 re-solves from scratch (not a repair)** | **UNRECORDED** | `unittest/v1` | 1 | `24` | **`0`** | `0` | `+0` |

> [!WARNING]
> **Unrecorded receipts** — `R2. 3.7-Flash solo (thinking=medium)`, `R3. 3.7-Flash solo (thinking=high)`, `R4. Gemini ladder: Lite -> 3.7(low) -> 3.7(medium)`, `R6. Gemini ladder -> Opus-5 (only after every Gemini rung fails)`, `R7. 3.7(medium) -> Opus-5 on the first failure (aggressive)`, `R9. Gemini ladder -> Opus-5 when the digest says the failure is hard`, `R10. Gemini ladder -> Opus-5 re-solves from scratch (not a repair)` captured runs through the harness but recorded no evidence treatment, so their `Sent`/`Native`/`Δ` columns are missing measurements rather than zeros. Do not read them as a result.

---

## 3. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `R6. Gemini ladder -> Opus-5 (only after every Gemini rung fails)` is the cheapest per solved task at `$0.0024`; `R1. 3.7-Flash solo (thinking=low)` has the highest pass rate at 100%.
