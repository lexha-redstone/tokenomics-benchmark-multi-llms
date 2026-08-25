# Comparative TCO Report: `straitjacket` on `FeatureBench` (N=48)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **FeatureBench** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **F4. Diff-Contract: Flash low -> Sonnet-5 (Strict unified diff anchoring)** | `Gemini 3.7 Flash (low) -> Claude Sonnet-5 (contracted diffs)` | Straitjacket contained digest ($0.00) | 2/48 (4.2%) | **2/48 (4.2%)** | `$4.1069` | `$0.0000` | **`$2.0535`** | `6069.1` |
| **F5. Diff-Aware Evidence Gate: Flash low -> Sonnet-5 / Opus-5 on hard/stalled** | `Gemini 3.7 Flash -> Claude Sonnet-5 / Opus-5 (diff-aware gate)` | Straitjacket digest + diff-aware escalation ($0.00) | 5/47 (10.6%) | **5/47 (10.6%)** | `$3.9609` | `$0.0000` | **`$0.7922`** | `6046.2` |
| **F6. Spec Deconstruct: Manifest extraction + Flash low synthesis & repair** | `Gemini 3.7 Flash manifest + Flash/Sonnet diff synthesis` | Straitjacket contained digest ($0.00) | 4/48 (8.3%) | **4/48 (8.3%)** | `$4.6103` | `$0.0000` | **`$1.1526`** | `5886.2` |

---

## 2. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **F4. Diff-Contract: Flash low -> Sonnet-5 (Strict unified diff anchoring)** | straitjacket | `pytest/v1, pytest/v2` | 6 | `86,571` | **`1,670`** | `1,772` | `+102` (+6%) |
| **F5. Diff-Aware Evidence Gate: Flash low -> Sonnet-5 / Opus-5 on hard/stalled** | straitjacket | `pytest/v1, pytest/v2` | 8 | `121,971` | **`1,654`** | `1,611` | `-43` (-3%) |
| **F6. Spec Deconstruct: Manifest extraction + Flash low synthesis & repair** | straitjacket | `pytest/v1, pytest/v2` | 8 | `179,162` | **`1,716`** | `1,778` | `+62` (+3%) |

---

## 3. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `F5. Diff-Aware Evidence Gate: Flash low -> Sonnet-5 / Opus-5 on hard/stalled` is the cheapest per solved task at `$0.7922`; `F5. Diff-Aware Evidence Gate: Flash low -> Sonnet-5 / Opus-5 on hard/stalled` has the highest pass rate at 11%.
