# Model Catalog Dimensions: Routing Beyond Raw Price

While capability tiers gate models and token prices break ties by default, empirical benchmark receipts reveal that non-price dimensions often dictate real-world performance, latency, and actual billed costs.

---

## 1. The Provenance Rule

> **Invariant**: Every quantitative claim in the catalog must carry a verified `source`. Unverified or fabricated numbers are strictly prohibited.

- **Absent Data = UNKNOWN (Never Bad)**: A model without throughput measurements is considered unmeasured, not slow. Unmeasured latency defaults to `moderate`, never optimistically to `fast`.
- **Declared Heuristic**: `declared-heuristic` is a valid source tag indicating expert judgment, which can be superseded by empirical benchmarks at any time.

---

## 2. Multi-Dimensional Routing Axes

| Dimension | Definition | Routing Impact |
|:---|:---|:---|
| **`specialities`** | Explicit domains where the model excels (e.g., `["reason", "architect", "synthesize"]`). | Primary tie-break among models clearing the capability tier. |
| **`anti_specialities`** | Declared weak areas (e.g., `["unbounded_log_analysis"]`). | Strong penalty during role matching; never a hard block. |
| **`latency_class`** | Perceived time-to-first-token (`fast`, `moderate`, `deliberate`). | Critical for interactive or high fan-out exploration nodes. |
| **`throughput_tok_s`** | **Empirically measured** generation speed (output tokens/sec). | Critical for large diff generation or high-volume code synthesis. |
| **`observed_behaviour`**| Verifiable behavioral traits captured in benchmark logs. | Decisive factor for containment and model allocation. |

---

## 3. Empirical Receipts & Critical Findings

### 1. Flood Discipline Splits by Model (Not Tier)
- **`gemini-3.5-flash-lite` (Low Flood Discipline)**:
  - On a greppable log task without strict containment, repeated log dumps 27 times, emitting **7.8 MB of raw text and 1.5M tool-output tokens**.
  - *Rule*: Route flood-prone or noisy tasks to Flash-lite **only behind Straitjacket containment (`ctx run`)**.
- **`gemini-3.6-flash` (High Flood Discipline)**:
  - On the exact same task, autonomously ran targeted `grep`, emitting only **812 bytes total**.

### 2. Unit Price vs. Context Expansion (Context Drag)
- **The Pitfall**: The cheapest unit price per token does not always yield the lowest total cost.
- **Evidence**: On a 3-phase web development task, an agent using `gemini-3.6-flash` re-sent **4.25M cumulative prompt tokens** across turns to produce 63k output tokens, narrowing the cost gap with Claude Sonnet to just $0.34.
- *Rule*: Minimize multi-turn context accumulation via prompt prefix stabilization and CAS checkpointing.

### 3. Quality Parity via Plan/Build Disaggregation
- **Opus Solo vs. Hybrid Routing**:
  - `Solo Claude Opus`: 98% pass rate @ **$9.59**.
  - `Opus Plan + Sonnet/Gemini Build`: 98% pass rate @ **$7.04** (1.43× cheaper).
- *Rule*: Invest flagship tokens in upfront specification and architecture; execute bulk code generation on standard/economy tiers.

### 4. Measured Throughput Data (Median, n=8)
- `gemini-3.6-flash`: **91.3 output tok/s**
- `gemini-3.5-flash-lite`: **58.8 output tok/s**
- *Insight*: The economy model is ~36% slower in sustained token generation than the standard model.
