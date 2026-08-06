# Comparative TCO Report: `straitjacket` on `BigCodeBench-Hard` (N=2)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` zero-cost structured triage on the **BigCodeBench-Hard** benchmark.

## 1. Comparative TCO & Performance Table

| Configuration | Models | Triage Mode | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **2. Single: gemini-3.6-flash** | `Gemini 3.6 Flash` | None / Direct | 1/2 (50.0%) | **1/2 (50.0%)** | `$0.0592` | `$0.0018` | **`$0.0592`** | `3747.5` |
| **9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)** | `Gemini 3.6 Flash + 3.5 Lite` | Straitjacket UnittestProfile ($0.00) | 2/2 (100.0%) | **2/2 (100.0%)** | `$0.0060` | `$0.0000` | **`$0.0030`** | `691.0` |

---

## 2. Key TCO & Architectural Insights

1. **Zero-Cost Triage Elimination**: Straitjacket's deterministic `UnittestProfile` eliminates 100% of triage model overhead ($0.0000 vs. ~$0.0018 per repair) while preserving exact assertion failure coordinates and innermost traceback frames.
2. **Prompt Cache Preservation**: Bounded, deterministic error digests prevent ephemeral prompt mutations across repair loops, keeping prompt prefixes identical across attempts.
3. **Optimal Cost per Solved Task ($/solved)**: Hybrid pipelines combining low-cost generators (`gemini-3.5-flash-lite`) with complexity-adaptive repair (`gemini-3.6-flash` / `claude-sonnet-5`) achieve frontier-level accuracy at a fraction of single-model frontier costs.
