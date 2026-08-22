# Benchmark Reports

Every sweep this repository has run, in execution order. The number is the
order, not a version: reports are append-only, so a later run never
overwrites an earlier one's evidence.

Regenerate this index with `python3 tools/index_reports.py --apply`.

| # | Date | Dataset | Tasks | Report | Dashboard | What it was |
|---|---|---|---|---|---|---|
| 01 | 2026-07-09 | BigCodeBench-Hard | N=30 | [md](01_bcb-hard_sweetspot_n30.md) | [html](01_bcb-hard_sweetspot_n30.html) | First sweet-spot sweep. gemini-3.1/3.5 + claude-opus-4, pre-straitjacket. |
| 02 | 2026-07-09 | BigCodeBench-Hard | N=30 | — | [html](02_bcb-hard_sweetspot_n30_rev.html) | Revision of the 01 dashboard (opus-4 arm dropped). |
| 03 | 2026-07-13 | WebDev | - | — | [html](03_webdev_sweetspot.html) | WebDev sweet-spot dashboard, pre-straitjacket. |
| 04 | inferred | BigCodeBench-Hard | N=10 | [md](04_bcb-hard_straitjacket_n10.md) | — | First straitjacket pilot. |
| 05 | inferred | BigCodeBench-Hard | N=30 | [md](05_bcb-hard_straitjacket_n30.md) | [html](05_bcb-hard_straitjacket_n30.html) | Straitjacket comparative TCO across all arms. |
| 06 | inferred | BigCodeBench-Hard | N=50 | [md](06_bcb-hard_gemini-vs-claude_n50.md) | — | Gemini vs Claude head-to-head. |
| 07 | inferred | SWE-bench Pro | N=10 | [md](07_swebench-pro_straitjacket_n10.md) | [html](07_swebench-pro_straitjacket_n10.html) | Straitjacket pilot on git-patch resolution. |
| 08 | inferred | SWE-bench Pro | N=30 | [md](08_swebench-pro_straitjacket_n30.md) | [html](08_swebench-pro_straitjacket_n30.html) | Straitjacket comparative TCO. |
| 09 | inferred | SWE-bench Pro | N=30 | [md](09_swebench-pro_live-api_n30.md) | [html](09_swebench-pro_live-api_n30.html) | Live Vertex AI run; adds claude-opus-4.8. |
| 10 | inferred | WebDev | N=2 | [md](10_webdev_straitjacket_n2.md) | [html](10_webdev_straitjacket_n2.html) | Straitjacket smoke run on web/networking tasks. |
| 11 | 2026-08-06 | cross-dataset | N=10/30/50 | [md](11_synthesis_cross-dataset.md) | — | Synthesis of every sweep up to that date. |

`inferred` dates: exact run timestamps were not recorded for those sweeps.
Their order is derived from task count and model generation, which is why it
is labelled rather than stated as fact.

Methodology documents live in [`../docs/`](../docs/), not here — this
directory holds run results only.
