# Comparative TCO Report: `straitjacket` on `FeatureBench` (N=48)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **FeatureBench** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **F0a. Single: gemini-3.7-flash low (2 rungs)** | `Gemini 3.7 Flash (low) x2` | Straitjacket contained digest ($0.00) | 6/48 (12.5%) | **6/48 (12.5%)** | `$5.0284` | `$0.0000` | **`$0.8381`** | `9610.1` |
| **F0b. Single: claude-sonnet-5 (2 rungs)** | `Claude Sonnet-5 x2` | Straitjacket contained digest ($0.00) | 2/48 (4.2%) | **2/48 (4.2%)** | `$8.1002` | `$0.0000` | **`$4.0501`** | `10303.2` |
| **F1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate)** | `Gemini 3.7 Flash -> Claude Sonnet-5` | Straitjacket contained digest ($0.00) | 7/48 (14.6%) | **7/48 (14.6%)** | `$9.8951` | `$0.0000` | **`$1.4136`** | `9239.2` |
| **F2. Evidence gate: same tiers, escalate when the digest says hard** | `Gemini 3.7 Flash -> Claude Sonnet-5 / Opus-5 (evidence gate)` | Straitjacket digest + evidence-gated escalation ($0.00) | 3/48 (6.2%) | **3/48 (6.2%)** | `$8.9630` | `$0.0000` | **`$2.9877`** | `8630.7` |
| **F3. H2: opus-5 plans first, 3.7-flash implements and repairs** | `Claude Opus-5 plan + Gemini 3.7 Flash exec x2` | Straitjacket contained digest ($0.00) | 1/48 (2.1%) | **1/48 (2.1%)** | `$8.5323` | `$0.0000` | **`$8.5323`** | `9349.0` |

---

## 2. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **F0a. Single: gemini-3.7-flash low (2 rungs)** | straitjacket | `pytest/v1` | 19 | `101,118` | **`4,251`** | `5,288` | `+1,037` (+20%) |
| **F0b. Single: claude-sonnet-5 (2 rungs)** | straitjacket | `pytest/v1, pytest/v2` | 14 | `133,227` | **`6,772`** | `5,515` | `-1,257` (-23%) |
| **F1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate)** | straitjacket | `pytest/v1, pytest/v2` | 21 | `304,996` | **`4,780`** | `5,265` | `+485` (+9%) |
| **F2. Evidence gate: same tiers, escalate when the digest says hard** | straitjacket | `pytest/v1, pytest/v2` | 12 | `149,605` | **`5,847`** | `4,716` | `-1,131` (-24%) |
| **F3. H2: opus-5 plans first, 3.7-flash implements and repairs** | straitjacket | `pytest/v1, pytest/v2` | 6 | `31,575` | **`1,838`** | `1,923` | `+85` (+4%) |

---

## 3. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `F0a. Single: gemini-3.7-flash low (2 rungs)` is the cheapest per solved task at `$0.8381`; `F1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate)` has the highest pass rate at 15%.
