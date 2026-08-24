# Comparative TCO Report: `straitjacket` on `ClassEval` (N=91)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **ClassEval** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **C0a. Single: gemini-3.5-flash-lite (whole class)** | `Gemini 3.5 Flash-Lite` | Straitjacket ($0.00) | 56/91 (61.5%) | **56/91 (61.5%)** | `$0.4040` | `$0.0000` | **`$0.0072`** | `1592.0` |
| **C0b. Single: gemini-3.7-flash low (whole class)** | `Gemini 3.7 Flash` | Straitjacket ($0.00) | 70/91 (76.9%) | **70/91 (76.9%)** | `$2.1593` | `$0.0000` | **`$0.0308`** | `2913.5` |
| **C0c. Single: claude-sonnet-5 (whole class)** | `Claude Sonnet-5` | Straitjacket ($0.00) | 66/91 (72.5%) | **66/91 (72.5%)** | `$1.9332` | `$0.0000` | **`$0.0293`** | `1778.4` |
| **C1. Cascade: whole class, Lite -> 3.7 low -> 3.7 medium** | `Gemini 3.5 Lite -> 3.7 Flash` | Straitjacket ($0.00) | 73/91 (80.2%) | **73/91 (80.2%)** | `$2.7094` | `$0.0000` | **`$0.0371`** | `4362.6` |
| **C2. Plan & execute: 3.7 contracts -> Lite writes class** | `Gemini 3.7 Flash plan + 3.5 Lite exec` | Straitjacket ($0.00) | 70/91 (76.9%) | **70/91 (76.9%)** | `$1.8594` | `$0.0000` | **`$0.0266`** | `3025.3` |
| **C3. CONTROL: per-method, every method to Lite** | `Gemini 3.5 Lite per method` | Straitjacket ($0.00) | 66/91 (72.5%) | **66/91 (72.5%)** | `$1.3886` | `$0.0000` | **`$0.0210`** | `2205.3` |
| **C4. H1: per-method, routed by labelled difficulty tier** | `Lite (easy) / 3.7 Flash (hard) per method` | Straitjacket ($0.00) | 65/91 (71.4%) | **65/91 (71.4%)** | `$2.0615` | `$0.0000` | **`$0.0317`** | `2715.9` |
| **C5. H1+plan: 3.7 contracts, then routed per-method execution** | `3.7 plan + Lite/3.7 routed per method` | Straitjacket ($0.00) | 65/91 (71.4%) | **65/91 (71.4%)** | `$2.6631` | `$0.0000` | **`$0.0410`** | `3235.8` |
| **C0d. Single: claude-opus-5 (whole class)** | `Claude Opus-5` | Straitjacket ($0.00) | 80/91 (87.9%) | **80/91 (87.9%)** | `$3.7120` | `$0.0000` | **`$0.0464`** | `1394.4` |

---

## 2. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **C0a. Single: gemini-3.5-flash-lite (whole class)** | straitjacket | `unittest/v1` | 743 | `160,500` | **`13,062`** | `22,349` | `+9,287` (+42%) |
| **C0b. Single: gemini-3.7-flash low (whole class)** | straitjacket | `text/v1, unittest/v1` | 658 | `96,061` | **`9,222`** | `15,289` | `+6,067` (+40%) |
| **C0c. Single: claude-sonnet-5 (whole class)** | straitjacket | `unittest/v1` | 702 | `124,451` | **`10,869`** | `19,221` | `+8,352` (+43%) |
| **C1. Cascade: whole class, Lite -> 3.7 low -> 3.7 medium** | straitjacket | `text/v1, unittest/v1` | 820 | `138,421` | **`17,467`** | `28,404` | `+10,937` (+39%) |
| **C2. Plan & execute: 3.7 contracts -> Lite writes class** | straitjacket | `text/v1, unittest/v1` | 689 | `132,182` | **`10,964`** | `18,969` | `+8,005` (+42%) |
| **C3. CONTROL: per-method, every method to Lite** | straitjacket | `text/v1, unittest/v1` | 716 | `139,392` | **`12,085`** | `20,345` | `+8,260` (+41%) |
| **C4. H1: per-method, routed by labelled difficulty tier** | straitjacket | `unittest/v1` | 674 | `101,137` | **`9,468`** | `16,321` | `+6,853` (+42%) |
| **C5. H1+plan: 3.7 contracts, then routed per-method execution** | straitjacket | `text/v1, unittest/v1` | 663 | `114,066` | **`9,738`** | `16,039` | `+6,301` (+39%) |
| **C0d. Single: claude-opus-5 (whole class)** | straitjacket | `unittest/v1` | 551 | `54,140` | **`4,310`** | `7,269` | `+2,959` (+41%) |

---

## 3. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `C0a. Single: gemini-3.5-flash-lite (whole class)` is the cheapest per solved task at `$0.0072`; `C0d. Single: claude-opus-5 (whole class)` has the highest pass rate at 88%.
