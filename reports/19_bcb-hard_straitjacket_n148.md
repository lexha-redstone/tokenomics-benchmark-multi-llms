# Comparative TCO Report: `straitjacket` on `BigCodeBench-Hard` (N=148)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **BigCodeBench-Hard** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **R0a. Baseline: claude-sonnet-5 solo (3 rungs)** | `Claude Sonnet-5 x3` | Straitjacket contained digest ($0.00) | 99/148 (66.9%) | **99/148 (66.9%)** | `$2.8619` | `$0.0000` | **`$0.0289`** | `1496.1` |
| **R0b. Baseline: claude-opus-5 solo (3 rungs)** | `Claude Opus-5 x3` | Straitjacket contained digest ($0.00) | 125/148 (84.5%) | **125/148 (84.5%)** | `$5.6972` | `$0.0000` | **`$0.0456`** | `1221.4` |
| **R1. 3.7-Flash solo (thinking=low)** | `Gemini 3.7 Flash (low) x3` | Straitjacket contained digest ($0.00) | 106/148 (71.6%) | **106/148 (71.6%)** | `$4.2716` | `$0.0000` | **`$0.0403`** | `3512.4` |
| **R2. 3.7-Flash solo (thinking=medium)** | `Gemini 3.7 Flash (medium) x3` | Straitjacket contained digest ($0.00) | 111/148 (75.0%) | **111/148 (75.0%)** | `$7.5978` | `$0.0000` | **`$0.0684`** | `6513.3` |
| **R4. Gemini ladder: Lite -> 3.7(low) -> 3.7(medium)** | `Lite -> 3.7 Flash low -> 3.7 Flash medium` | Straitjacket contained digest ($0.00) | 108/148 (73.0%) | **108/148 (73.0%)** | `$3.9043` | `$0.0000` | **`$0.0362`** | `3666.2` |
| **R5. Gemini thinking ladder: 3.7 low -> medium -> high (no Lite)** | `3.7 Flash low -> medium -> high` | Straitjacket contained digest ($0.00) | 112/148 (75.7%) | **112/148 (75.7%)** | `$7.5467` | `$0.0000` | **`$0.0674`** | `6458.3` |
| **R6. Gemini ladder -> Opus-5 (only after every Gemini rung fails)** | `Lite -> 3.7 low -> 3.7 medium -> Opus-5` | Straitjacket contained digest ($0.00) | 125/148 (84.5%) | **125/148 (84.5%)** | `$5.6533` | `$0.0000` | **`$0.0452`** | `4269.3` |
| **R7. 3.7(medium) -> Opus-5 on the first failure (aggressive)** | `3.7 Flash medium -> Opus-5` | Straitjacket contained digest ($0.00) | 112/148 (75.7%) | **112/148 (75.7%)** | `$5.5787` | `$0.0000` | **`$0.0498`** | `3152.5` |
| **R8. 3.7 low -> medium -> Opus-5 after two failures** | `3.7 low -> 3.7 medium -> Opus-5` | Straitjacket contained digest ($0.00) | 125/148 (84.5%) | **125/148 (84.5%)** | `$5.8400` | `$0.0000` | **`$0.0467`** | `3858.4` |
| **R9. Gemini ladder -> Opus-5 when the digest says the failure is hard** | `Lite -> 3.7 low -> 3.7 medium -> Opus-5 (evidence gate)` | Straitjacket digest + evidence-gated escalation ($0.00) | 120/148 (81.1%) | **120/148 (81.1%)** | `$4.2374` | `$0.0000` | **`$0.0353`** | `2615.5` |
| **R10. Gemini ladder -> Opus-5 re-solves from scratch (not a repair)** | `Lite -> 3.7 low -> 3.7 medium -> Opus-5 (fresh solve)` | Straitjacket contained digest ($0.00) | 122/148 (82.4%) | **122/148 (82.4%)** | `$5.0849` | `$0.0000` | **`$0.0417`** | `4046.0` |

---

## 2. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **R0a. Baseline: claude-sonnet-5 solo (3 rungs)** | straitjacket | `text/v1, unittest/v1` | 278 | `115,986` | **`25,414`** | `55,370` | `+29,956` (+54%) |
| **R0b. Baseline: claude-opus-5 solo (3 rungs)** | straitjacket | `text/v1, unittest/v1` | 225 | `62,569` | **`15,035`** | `31,205` | `+16,170` (+52%) |
| **R1. 3.7-Flash solo (thinking=low)** | straitjacket | `text/v1, unittest/v1` | 289 | `110,220` | **`26,544`** | `55,729` | `+29,185` (+52%) |
| **R2. 3.7-Flash solo (thinking=medium)** | straitjacket | `text/v1, unittest/v1` | 286 | `104,206` | **`26,430`** | `56,731` | `+30,301` (+53%) |
| **R4. Gemini ladder: Lite -> 3.7(low) -> 3.7(medium)** | straitjacket | `text/v1, unittest/v1` | 291 | `113,426` | **`28,494`** | `59,767` | `+31,273` (+52%) |
| **R5. Gemini thinking ladder: 3.7 low -> medium -> high (no Lite)** | straitjacket | `text/v1, unittest/v1` | 288 | `105,878` | **`26,712`** | `54,877` | `+28,165` (+51%) |
| **R6. Gemini ladder -> Opus-5 (only after every Gemini rung fails)** | straitjacket | `text/v1, unittest/v1` | 335 | `127,837` | **`36,529`** | `76,598` | `+40,069` (+52%) |
| **R7. 3.7(medium) -> Opus-5 on the first failure (aggressive)** | straitjacket | `text/v1, unittest/v1` | 232 | `74,673` | **`16,549`** | `33,917` | `+17,368` (+51%) |
| **R8. 3.7 low -> medium -> Opus-5 after two failures** | straitjacket | `text/v1, unittest/v1` | 286 | `100,626` | **`26,642`** | `54,436` | `+27,794` (+51%) |
| **R9. Gemini ladder -> Opus-5 when the digest says the failure is hard** | straitjacket | `text/v1, unittest/v1` | 295 | `104,966` | **`28,447`** | `58,153` | `+29,706` (+51%) |
| **R10. Gemini ladder -> Opus-5 re-solves from scratch (not a repair)** | straitjacket | `text/v1, unittest/v1` | 330 | `128,647` | **`34,726`** | `71,605` | `+36,879` (+52%) |

---

## 3. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `R0a. Baseline: claude-sonnet-5 solo (3 rungs)` is the cheapest per solved task at `$0.0289`; `R0b. Baseline: claude-opus-5 solo (3 rungs)` has the highest pass rate at 84%.
