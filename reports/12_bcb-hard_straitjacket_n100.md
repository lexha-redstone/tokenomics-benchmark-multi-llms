# Comparative TCO Report: `straitjacket` on `BigCodeBench-Hard` (N=100)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **BigCodeBench-Hard** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **2. Single: gemini-3.7-flash** | `Gemini 3.7 Flash` | Straitjacket digest ($0.00, repair turn) | 60/100 (60.0%) | **60/100 (60.0%)** | `$2.2301` | `$0.0000` | **`$0.0372`** | `2720.3` |
| **3. Single: claude-sonnet-5** | `Claude Sonnet-5` | Straitjacket digest ($0.00, repair turn) | 54/100 (54.0%) | **54/100 (54.0%)** | `$1.5366` | `$0.0000` | **`$0.0285`** | `1215.1` |
| **4. Single: claude-opus-5** | `Claude Opus-5` | Straitjacket digest ($0.00, repair turn) | 76/100 (76.0%) | **76/100 (76.0%)** | `$3.5162` | `$0.0000` | **`$0.0463`** | `1135.7` |
| **8. Straitjacket Cascade (3.5-Lite -> 3.7-Flash)** | `Gemini 3.5 Lite -> 3.7 Flash` | Straitjacket contained digest ($0.00) | 66/100 (66.0%) | **66/100 (66.0%)** | `$2.2349` | `$0.0000` | **`$0.0339`** | `3109.6` |
| **9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)** | `Gemini 3.7 Flash + 3.5 Lite` | Straitjacket contained digest ($0.00) | 59/100 (59.0%) | **59/100 (59.0%)** | `$1.6328` | `$0.0000` | **`$0.0277`** | `2249.1` |
| **11. Straitjacket Smart Repair (Pure Gemini)** | `Gemini 3.7 Flash -> 3.5 Lite -> Flash (Med)` | Straitjacket contained digest ($0.00) | 64/100 (64.0%) | **64/100 (64.0%)** | `$2.7209` | `$0.0000` | **`$0.0425`** | `3626.0` |
| **10. Straitjacket Escalation Shield (Gemini -> Claude)** | `Gemini Lite -> Flash -> Claude Sonnet-5` | Straitjacket contained digest ($0.00) | 68/100 (68.0%) | **68/100 (68.0%)** | `$1.9162` | `$0.0000` | **`$0.0282`** | `2481.8` |

---

## 2. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **2. Single: gemini-3.7-flash** | straitjacket | `text/v1, unittest/v1` | 165 | `68,671` | **`12,291`** | `26,169` | `+13,878` (+53%) |
| **3. Single: claude-sonnet-5** | straitjacket | `text/v1, unittest/v1` | 158 | `59,960` | **`10,939`** | `22,969` | `+12,030` (+52%) |
| **4. Single: claude-opus-5** | straitjacket | `text/v1, unittest/v1` | 140 | `35,756` | **`8,024`** | `16,513` | `+8,489` (+51%) |
| **8. Straitjacket Cascade (3.5-Lite -> 3.7-Flash)** | straitjacket | `text/v1, unittest/v1` | 206 | `80,009` | **`20,892`** | `43,845` | `+22,953` (+52%) |
| **9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)** | straitjacket | `text/v1, unittest/v1` | 163 | `66,801` | **`12,420`** | `26,142` | `+13,722` (+52%) |
| **11. Straitjacket Smart Repair (Pure Gemini)** | straitjacket | `text/v1, unittest/v1` | 224 | `99,103` | **`24,328`** | `52,925` | `+28,597` (+54%) |
| **10. Straitjacket Escalation Shield (Gemini -> Claude)** | straitjacket | `text/v1, unittest/v1` | 209 | `80,828` | **`20,799`** | `42,807` | `+22,008` (+51%) |

---

## 3. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)` is the cheapest per solved task at `$0.0277`; `4. Single: claude-opus-5` has the highest pass rate at 76%.
