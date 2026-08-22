# Comparative TCO Report: `straitjacket` on `BigCodeBench-Hard` (N=100)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **BigCodeBench-Hard** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **2. Single: gemini-3.7-flash** | `Gemini 3.7 Flash` | Straitjacket digest ($0.00, repair turn) | 62/100 (62.0%) | **62/100 (62.0%)** | `$2.2853` | `$0.0000` | **`$0.0369`** | `2792.3` |
| **3. Single: claude-sonnet-5** | `Claude Sonnet-5` | Straitjacket digest ($0.00, repair turn) | 53/100 (53.0%) | **53/100 (53.0%)** | `$1.5733` | `$0.0000` | **`$0.0297`** | `1239.7` |
| **4. Single: claude-opus-5** | `Claude Opus-5` | Straitjacket digest ($0.00, repair turn) | 72/100 (72.0%) | **72/100 (72.0%)** | `$3.6106` | `$0.0000` | **`$0.0501`** | `1174.3` |
| **8. Straitjacket Cascade (3.5-Lite -> 3.7-Flash)** | `Gemini 3.5 Lite -> 3.7 Flash` | Straitjacket contained digest ($0.00) | 62/100 (62.0%) | **62/100 (62.0%)** | `$2.1754` | `$0.0000` | **`$0.0351`** | `3023.2` |
| **9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)** | `Gemini 3.7 Flash + 3.5 Lite` | Straitjacket contained digest ($0.00) | 59/100 (59.0%) | **59/100 (59.0%)** | `$1.6206` | `$0.0000` | **`$0.0275`** | `2253.2` |
| **11. Straitjacket Smart Repair (Pure Gemini)** | `Gemini 3.7 Flash -> 3.5 Lite -> Flash (Med)` | Straitjacket contained digest ($0.00) | 62/100 (62.0%) | **62/100 (62.0%)** | `$2.7858` | `$0.0000` | **`$0.0449`** | `3707.8` |
| **10. Straitjacket Escalation Shield (Gemini -> Claude)** | `Gemini Lite -> Flash -> Claude Sonnet-5` | Straitjacket contained digest ($0.00) | 64/100 (64.0%) | **64/100 (64.0%)** | `$1.8102` | `$0.0000` | **`$0.0283`** | `2361.1` |

---

## 2. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **2. Single: gemini-3.7-flash** | straitjacket | `text/v1, unittest/v1` | 164 | `58,809` | **`13,028`** | `25,623` | `+12,595` (+49%) |
| **3. Single: claude-sonnet-5** | straitjacket | `text/v1, unittest/v1` | 161 | `64,752` | **`11,482`** | `23,555` | `+12,073` (+51%) |
| **4. Single: claude-opus-5** | straitjacket | `text/v1, unittest/v1` | 139 | `39,428` | **`7,976`** | `16,805` | `+8,829` (+53%) |
| **8. Straitjacket Cascade (3.5-Lite -> 3.7-Flash)** | **UNRECORDED** | `text/v1, unittest/v1` | 211 | `85,886` | **`0`** | `0` | `+0` |
| **9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)** | **UNRECORDED** | `text/v1, unittest/v1` | 165 | `63,885` | **`0`** | `0` | `+0` |
| **11. Straitjacket Smart Repair (Pure Gemini)** | **UNRECORDED** | `text/v1, unittest/v1` | 221 | `101,791` | **`0`** | `0` | `+0` |
| **10. Straitjacket Escalation Shield (Gemini -> Claude)** | **UNRECORDED** | `text/v1, unittest/v1` | 207 | `90,811` | **`0`** | `0` | `+0` |

> [!WARNING]
> **Unrecorded receipts** — `8. Straitjacket Cascade (3.5-Lite -> 3.7-Flash)`, `9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)`, `11. Straitjacket Smart Repair (Pure Gemini)`, `10. Straitjacket Escalation Shield (Gemini -> Claude)` captured runs through the harness but recorded no evidence treatment, so their `Sent`/`Native`/`Δ` columns are missing measurements rather than zeros. Do not read them as a result.

---

## 3. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)` is the cheapest per solved task at `$0.0275`; `4. Single: claude-opus-5` has the highest pass rate at 72%.
