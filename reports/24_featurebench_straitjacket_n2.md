# Comparative TCO Report: `straitjacket` on `FeatureBench` (N=2)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **FeatureBench** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **F1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate)** | `Gemini 3.7 Flash -> Claude Sonnet-5` | Straitjacket contained digest ($0.00) | 0/2 (0.0%) | **0/2 (0.0%)** | `$1.2194` | `$0.0000` | **`$0.0000`** | `26790.5` |
| **F7. Grounded cascade: repository quoted, Flash low -> Sonnet-5 (3 rungs)** | `Gemini 3.7 Flash (low) -> Claude Sonnet-5, source-grounded` | Straitjacket contained digest ($0.00) | 0/2 (0.0%) | **0/2 (0.0%)** | `$1.0859` | `$0.0000` | **`$0.0000`** | `22678.5` |

---

## 2. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **F1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate)** | straitjacket | `pytest/v2` | 3 | `8,319` | **`2,175`** | `2,217` | `+42` (+2%) |

---

## 3. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
