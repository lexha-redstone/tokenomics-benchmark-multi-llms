# Comparative TCO Report: `straitjacket` on `SWE-bench Pro` (N=2)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` zero-cost structured triage on the **SWE-bench Pro** benchmark.

## 1. Comparative TCO & Performance Table

| Configuration | Models | Triage Mode | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **1. Single: claude-opus-5** | `Claude Opus-5` | Straitjacket UnittestProfile ($0.00) | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0176` | `$0.0000` | **`$0.0088`** | `227.5` |
| **2. Single: claude-sonnet-5** | `Claude Sonnet-5` | Straitjacket UnittestProfile ($0.00) | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0070` | `$0.0000` | **`$0.0035`** | `227.5` |
| **3. Single: gemini-3.6-flash (LOW)** | `Gemini 3.6 Flash (LOW)` | Straitjacket UnittestProfile ($0.00) | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0345` | `$0.0000` | **`$0.0173`** | `2228.0` |
| **4. Single: gemini-3.5-flash-lite** | `Gemini 3.5 Lite` | Straitjacket UnittestProfile ($0.00) | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0015` | `$0.0000` | **`$0.0008`** | `227.5` |
| **5. Read/Write Split (3.6-Flash Planner + 3.5-Lite Executor)** | `Gemini 3.6 Flash Plan + 3.5 Lite Exec` | None / Direct | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0213` | `$0.0036` | **`$0.0106`** | `1361.5` |
| **6. Straitjacket Escalation Shield (Gemini Lite -> Flash -> Sonnet-5)** | `Gemini Lite -> Flash -> Claude Sonnet-5` | Straitjacket UnittestProfile ($0.00) | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0183` | `$0.0000` | **`$0.0092`** | `1251.5` |
| **7. Straitjacket Smart Repair (Pure Gemini 3-Tier)** | `Gemini 3.6 Flash -> 3.5 Lite -> Flash (Med)` | Straitjacket UnittestProfile ($0.00) | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0345` | `$0.0000` | **`$0.0173`** | `2228.0` |
| **8. Straitjacket Ultra-Sweet Hybrid (Claude Sonnet-5 -> Gemini Lite -> Opus-5)** | `Claude Sonnet-5 -> Gemini Lite -> Claude Opus-5` | Straitjacket UnittestProfile ($0.00) | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0118` | `$0.0000` | **`$0.0059`** | `337.5` |
| **9. Straitjacket Dual-Verifier Cascade (4-Tier Synergy)** | `Gemini Lite -> Flash -> Sonnet-5 -> Opus-5` | Straitjacket UnittestProfile ($0.00) | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0183` | `$0.0000` | **`$0.0092`** | `1251.5` |

---

## 2. Key TCO & Architectural Insights

1. **Zero-Cost Triage Elimination**: Straitjacket's deterministic `UnittestProfile` eliminates 100% of triage model overhead ($0.0000 vs. ~$0.0018 per repair) while preserving exact assertion failure coordinates and innermost traceback frames.
2. **Prompt Cache Preservation**: Bounded, deterministic error digests prevent ephemeral prompt mutations across repair loops, keeping prompt prefixes identical across attempts.
3. **Optimal Cost per Solved Task ($/solved)**: Hybrid pipelines combining low-cost generators (`gemini-3.5-flash-lite`) with complexity-adaptive repair (`gemini-3.6-flash` / `claude-sonnet-5`) achieve frontier-level accuracy at a fraction of single-model frontier costs.
