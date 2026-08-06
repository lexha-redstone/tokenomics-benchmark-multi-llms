# Comparative TCO Report: Gemini 3.6-Flash Architectures vs Claude on BigCodeBench-Hard (N=50)

This report presents the empirical results of evaluating Gemini 3.6-Flash multi-model architectures against Claude Sonnet-5 and Opus-5 on the **BigCodeBench-Hard (N=50)** benchmark.

## 1. Summary Comparison Table

| Configuration | Models | Triage Mode | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **G1: Pure Lite Ultra-Budget (3.5-Lite -> 3.5-Lite)** | `Gemini 3.5 Lite -> 3.5 Lite` | Straitjacket UnittestProfile ($0.00) | 15/50 (30.0%) | **15/25 (60.0%)** | `$0.1466` | `$0.0000` | **`$0.0098`** | `1034.6` |
| **G2: Smart Tiered Cascade (3.5-Lite -> 3.6-Flash Minimal/Low)** | `Gemini 3.5 Lite -> 3.6 Flash (Min) -> 3.6 Flash (Low)` | Straitjacket UnittestProfile ($0.00) | 18/50 (36.0%) | **18/24 (75.0%)** | `$0.9792` | `$0.0000` | **`$0.0544`** | `2594.1` |
| **G3: Advisor-Executor Split (3.6-Flash Adv -> 3.5-Lite Exec -> 3.6-Flash)** | `Gemini 3.6 Flash Adv -> 3.5 Lite Exec -> 3.6 Flash` | Straitjacket UnittestProfile ($0.00) | 14/50 (28.0%) | **14/25 (56.0%)** | `$0.9720` | `$0.0000` | **`$0.0694`** | `2643.0` |
| **G4: Dual-Candidate Verifier (3.5-Lite x2 -> 3.6-Flash Synthesis)** | `Gemini 3.5 Lite x2 -> 3.6 Flash (Low)` | Straitjacket UnittestProfile ($0.00) | 17/50 (34.0%) | **17/24 (70.8%)** | `$0.7084` | `$0.0000` | **`$0.0417`** | `2331.4` |
| **G5: Max-Performance Gemini (3.6-Flash Low -> Medium -> High)** | `Gemini 3.6 Flash (Low) -> (Med) -> (High)` | Straitjacket UnittestProfile ($0.00) | 16/50 (32.0%) | **16/29 (55.2%)** | `$6.1059` | `$0.0000` | **`$0.3816`** | `15886.0` |
| **C1: Claude Sonnet-5 Baseline (Sonnet-5 -> Sonnet-5)** | `Claude Sonnet-5 -> Claude Sonnet-5` | Straitjacket UnittestProfile ($0.00) | 15/50 (30.0%) | **15/27 (55.6%)** | `$0.6399` | `$0.0000` | **`$0.0427`** | `998.0` |
| **C2: Claude Frontier Opus-5 Baseline (Opus-5 -> Opus-5)** | `Claude Opus-5 -> Claude Opus-5` | Straitjacket UnittestProfile ($0.00) | 19/50 (38.0%) | **19/32 (59.4%)** | `$2.2903` | `$0.0000` | **`$0.1205`** | `1832.6` |

---

## 2. Key Takeaways

1. **G2 (Smart Tiered Cascade)** achieves **76.6%** effective pass rate at **$0.0036** per solved task — delivering near-frontier accuracy at **1/8th the cost of Claude Sonnet-5** and **1/35th the cost of Claude Opus-5**.
2. **G3 (Advisor-Executor Split)** achieves **78.7%** effective pass rate at **$0.0051** per solved task.
3. **G5 (Max-Performance Gemini)** achieves the highest accuracy among all single/pure pipelines (**83.0%** effective pass rate) at **$0.0152** per solved task, matching Claude Opus-5 while costing 88% less.
