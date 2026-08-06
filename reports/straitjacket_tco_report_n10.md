# Comparative TCO Report: `straitjacket` on `BigCodeBench-hard` (N=10 Pilot)

This report presents the empirical evaluation of `straitjacket` context containment on BigCodeBench-Hard (first 10 tasks).

## 1. TCO Comparison Table

| Arm | Architecture | Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|
| **Arm 0: Raw Stderr Baseline** | `cascade` | 3/10 (30%) | `$0.2108` | `$0.0000` | **`$0.0703`** | `2613.8` |
| **Arm 1: Shipped LLM Triage (Arch E)** | `hybrid` | 2/10 (20%) | `$0.1651` | `$0.0024` | **`$0.0825`** | `2365.3` |
| **Arm 2: Straitjacket Local Triage (Arch E-SJ)** | `hybrid-straitjacket` | 3/10 (30%) | `$0.1675` | `$0.0000` | **`$0.0558`** | `2282.6` |
| **Arm 3: Straitjacket Multi-Attempt Cache-Warmed (Arch C-SJ)** | `cascade-straitjacket` | 4/10 (40%) | `$0.3685` | `$0.0000` | **`$0.0921`** | `2179.5` |

---

## 2. Key Empirical Findings

1. **Zero-Cost Triage**: Replacing LLM-based triage (`triage_error`) with `straitjacket`'s local deterministic `UnittestProfile` eliminates 100% of the triage model spend while delivering structured failure coordinates.
2. **Cost per Solved Task ($/solved)**: Arm 2 (`Architecture E-SJ`) reduces the cost per solved task compared to both raw stderr baselines and shipped LLM triage.
3. **Prompt Cache Preservation**: Bounded, deterministic digests prevent prompt prefix mutations across multi-attempt repair loops.
