# Comparative TCO Report: `straitjacket` on `FeatureBench` (N=2)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **FeatureBench** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **F1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate)** | `Gemini 3.7 Flash -> Claude Sonnet-5` | Straitjacket contained digest ($0.00) | 0/2 (0.0%) | **0/2 (0.0%)** | `$1.2357` | `$0.0000` | **`N/A`** | `27399.0` |
| **F7. Grounded cascade: repository quoted, Flash low -> Sonnet-5 (3 rungs)** | `Gemini 3.7 Flash (low) -> Claude Sonnet-5, source-grounded` | Straitjacket contained digest ($0.00) | 0/2 (0.0%) | **0/2 (0.0%)** | `$1.3522` | `$0.0000` | **`N/A`** | `22982.5` |

---

## 2. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
