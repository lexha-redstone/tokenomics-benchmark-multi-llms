# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Unified Report and Interactive HTML Dashboard Generator for Multi-LLM Benchmarks.
Generates:
  1. Standard Markdown TCO Comparative Reports with Error Audits
  2. Modern Interactive HTML Dashboards with KPI Cards and Charts
"""

import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_HERE)

# ==============================================================================
# --- REPORT NAMING ---
# ==============================================================================
#
# Reports are an append-only, chronologically indexed log of sweeps:
#
#     reports/NN_<dataset>_<tag>_n<N>.md
#     reports/NN_<dataset>_<tag>_n<N>.html
#
# The index is the run order, so a reader can follow how the numbers evolved
# without cross-referencing git. A sweep's markdown and dashboard share one
# index. Fixed filenames were the old convention and meant every sweep
# silently overwrote the previous one's evidence.

REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

_DATASET_SLUG = {
    "bigcodebench-hard": "bcb-hard",
    "webdev": "webdev",
    "web-dev": "webdev",
    "featurebench": "featurebench",
    "feature-bench": "featurebench",
    "swe-bench pro": "swebench-pro",
    "swebench pro": "swebench-pro",
    "swebench-pro": "swebench-pro",
}

_INDEX_RE = re.compile(r"^(\d{2})_")


def dataset_slug(dataset_name):
    key = str(dataset_name).strip().lower()
    if key in _DATASET_SLUG:
        return _DATASET_SLUG[key]
    return re.sub(r"[^a-z0-9]+", "-", key).strip("-") or "dataset"


def next_report_index(reports_dir=None):
    """One past the highest index currently in the reports directory."""
    d = reports_dir or REPORTS_DIR
    highest = 0
    if os.path.isdir(d):
        for name in os.listdir(d):
            m = _INDEX_RE.match(name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


def allocate_report_paths(dataset_name, n_tasks, tag="straitjacket", reports_dir=None):
    """Reserve one chronological index for a sweep. Returns (md_path, html_path).

    Call once per sweep and pass the results to both generators, so the
    markdown and the dashboard carry the same index.
    """
    d = reports_dir or REPORTS_DIR
    os.makedirs(d, exist_ok=True)
    idx = next_report_index(d)
    stem = f"{idx:02d}_{dataset_slug(dataset_name)}_{tag}_n{n_tasks}"
    return os.path.join(d, stem + ".md"), os.path.join(d, stem + ".html")


def _harness_row(summary_rows):
    for r in summary_rows:
        sj = r.get("straitjacket")
        if sj:
            return sj
    return None


def _write_provenance(f, summary_rows):
    """State which harness produced the digests. A row labelled with a
    containment mechanism has to name the mechanism that actually ran."""
    sj = _harness_row(summary_rows)
    if not sj:
        return
    if sj.get("available"):
        f.write(f"> **Harness provenance** — digests produced by `ctx-harness` "
                f"v{sj.get('ctx_version')} via the `{sj.get('backend')}` backend "
                f"(upstream `ctx.digest` profile registry, unmodified). "
                f"Uncontained arms send at most {sj.get('raw_cap_chars')} chars of raw output "
                f"(`SJ_RAW_CAP`).\n\n")
    else:
        f.write("> [!WARNING]\n"
                f"> **Harness unavailable** ({sj.get('reason')}). No straitjacket arm in this "
                f"report was produced by the real harness.\n\n")


def _write_insights(f, summary_rows):
    """Only assert things about arms that are actually in this report.

    The insights used to be a fixed block naming an LLM-triage arm, an `A4`
    retrieval arm and `gemini-3.6-flash` regardless of what was run — so an
    N=100 sweep of seven other variants shipped three claims about rows that
    were not in the table.
    """
    treatments = {t for r in summary_rows
                  for t in ((r.get("containment") or {}).get("treatments") or [])}
    f.write("## 3. Key TCO & Architectural Insights\n\n")
    n = 0

    if "straitjacket" in treatments:
        n += 1
        f.write(f"{n}. **Containment, not compression**: the straitjacket arms send the "
                "harness's own digest for the failing run — profile-detected, "
                "coverage-attested, and carrying `ctx get` / `ctx search` addresses for "
                "every omitted region. No triage model is called, so their triage cost is "
                "$0.0000.\n")
    if "llm" in treatments:
        n += 1
        f.write(f"{n}. **Paid triage is the comparison, not the mechanism**: the LLM-triage "
                "arm buys the same brevity with a round trip and per-repair tokens; the "
                "Triage Cost column prices it.\n")
    if any((r.get("containment") or {}).get("captures") for r in summary_rows):
        n += 1
        f.write(f"{n}. **Residency, not just spend**: the containment table reports what each "
                "arm sent against what the untreated path would have sent. Dollars measure one "
                "turn; residency measures every turn those bytes would have stayed in the "
                "transcript. A negative delta is reported as readily as a positive one.\n")
        n += 1
        f.write(f"{n}. **Omission is not amnesia**: what the digest leaves out stays retrievable "
                "at an exact address, so a shorter prompt does not mean lost evidence.\n")
        n += 1
        f.write(f"{n}. **Where containment does nothing**: a run whose whole output is a handful "
                "of lines has nothing to contain, and its delta lands at or below zero. That is "
                "reported rather than hidden — short output is not automatically good output.\n")

    if summary_rows:
        best = min((r for r in summary_rows if r.get("passed")),
                   key=lambda r: r.get("cost_per_solved_usd", float("inf")),
                   default=None)
        top = max(summary_rows, key=lambda r: r.get("pass_rate", 0), default=None)
        if best and top:
            n += 1
            f.write(f"{n}. **Cost per solved task**: `{best.get('name')}` is the cheapest per "
                    f"solved task at `${best.get('cost_per_solved_usd', 0):.4f}`; "
                    f"`{top.get('name')}` has the highest pass rate at "
                    f"{top.get('pass_rate', 0):.0%}.\n")


def _write_containment_table(f, summary_rows):
    """Context-residency receipt per configuration."""
    rows = [r for r in summary_rows if (r.get("containment") or {}).get("captures")]
    if not rows:
        return
    f.write("\n---\n\n")
    f.write("## 2. Context Containment Receipt\n\n")
    f.write("Measured by the harness itself for every captured run in the sweep. Every arm "
            "executes through the harness; `Captured` differs between them because they "
            "make different numbers of attempts and their candidate solutions print "
            "different amounts. What the comparison turns on is which payload each arm "
            "put in front of the model.\n\n"
            "- **Captured** — everything the execution produced; the store holds all of it.\n"
            "- **Sent to model** — what this arm actually placed in the repair prompt.\n"
            "- **Native baseline** — what the *untreated* path would have forwarded for the "
            "same failures (the failing stream, tail-truncated).\n"
            "- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the "
            "treatment bought: an untreated harness also discards streams it never reads. "
            "The difference is that discarding is amnesia, while straitjacket's omissions "
            "are counted in a coverage receipt and remain retrievable by address.\n\n")
    f.write("| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | "
            "Native baseline | Δ vs native |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    blank_receipts = []
    for r in rows:
        c = r["containment"]
        raw = c.get("raw_tokens_est", 0)
        sent = c.get("evidence_sent_tokens_est", 0)
        native = c.get("native_baseline_tokens_est", 0)
        delta = native - sent
        pct = f" ({delta / native:+.0%})" if native else ""
        treatments = ", ".join(c.get("treatments") or [])
        if not treatments:
            # An arm with captures but no recorded treatment is an
            # instrumentation gap, not a zero-cost result. Say which.
            treatments = "**UNRECORDED**"
            blank_receipts.append(r.get("name", r.get("arm", "Variant")))
        f.write(f"| **{r.get('name', r.get('arm', 'Variant'))}** | "
                f"{treatments} | "
                f"`{', '.join(c.get('profiles') or []) or 'n/a'}` | "
                f"{c.get('captures', 0)} | "
                f"`{raw:,}` | "
                f"**`{sent:,}`** | "
                f"`{native:,}` | "
                f"`{delta:+,}`{pct} |\n")
    if blank_receipts:
        f.write("\n> [!WARNING]\n"
                "> **Unrecorded receipts** — " + ", ".join(f"`{n}`" for n in blank_receipts) +
                " captured runs through the harness but recorded no evidence treatment, so "
                "their `Sent`/`Native`/`Δ` columns are missing measurements rather than zeros. "
                "Do not read them as a result.\n")


def generate_markdown_report(summary_rows, dataset_name="BigCodeBench-Hard", output_path=None):
    """
    Generate standard Markdown Comparative TCO Report.
    """
    if not output_path:
        n_tasks = summary_rows[0].get("n", 0) if summary_rows else 0
        output_path = allocate_report_paths(dataset_name, n_tasks)[0]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n_tasks = summary_rows[0]["n"] if summary_rows else 30
    total_infra_errors = sum(r.get("infra_err_count", 0) for r in summary_rows)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Comparative TCO Report: `straitjacket` on `{dataset_name}` (N={n_tasks})\n\n")
        f.write(f"This report presents the empirical evaluation of multi-model collaboration architectures and "
                f"`straitjacket` context containment on the **{dataset_name}** benchmark.\n\n")
        _write_provenance(f, summary_rows)

        if total_infra_errors > 0:
            f.write("> [!WARNING]\n")
            f.write("> **Infrastructure & Environment Error Audit**: We audited all task failures and identified non-model "
                    "environment constraints (e.g. missing package imports like `ModuleNotFoundError` or Vertex AI quota limits). "
                    "Below we report both **Raw Pass Rate** and **Effective Pass Rate** (evaluating testable tasks).\n\n")

        f.write("## 1. Comparative TCO & Performance Table\n\n")
        f.write("| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        for r in summary_rows:
            name = r.get("name", r.get("arm", "Variant"))
            models = r.get("models", "N/A")
            triage_mode = r.get("triage_mode", "$0.00")
            n_val = r.get("n", n_tasks)
            passed = r.get("passed", 0)
            raw_pr = f"{passed}/{n_val} ({(passed/n_val)*100:.1f}%)" if n_val > 0 else "0.0%"
            
            infra_err = r.get("infra_err_count", 0)
            eff_denom = n_val - infra_err
            eff_pr = f"{passed}/{eff_denom} ({(passed/eff_denom)*100:.1f}%)" if eff_denom > 0 else raw_pr
            
            tot_usd = r.get("total_as_run_usd", r.get("total_usd", 0.0))
            triage_usd = r.get("total_triage_usd", r.get("triage_usd", 0.0))
            cps = r.get("cost_per_solved_usd", r.get("cost_per_solved", 0.0))
            avg_out = r.get("avg_output_tokens", r.get("avg_out_tok", 0.0))

            f.write(f"| **{name}** | `{models}` | {triage_mode} | {raw_pr} | **{eff_pr}** | `${tot_usd:.4f}` | `${triage_usd:.4f}` | **`${cps:.4f}`** | `{avg_out:.1f}` |\n")

        _write_containment_table(f, summary_rows)

        f.write("\n---\n\n")
        _write_insights(f, summary_rows)

    print(f"Generated Markdown report: {output_path}", flush=True)
    return output_path

def generate_html_dashboard(summary_rows, dataset_name="BigCodeBench-Hard", output_path=None):
    """
    Generate modern interactive HTML dashboard.
    """
    if not output_path:
        n_tasks = summary_rows[0].get("n", 0) if summary_rows else 0
        output_path = allocate_report_paths(dataset_name, n_tasks)[1]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sorted_rows = sorted(summary_rows, key=lambda x: (-x.get("pass_rate", 0), x.get("total_as_run_usd", 999)))
    best_perf = sorted_rows[0] if sorted_rows else {}
    # `n` is 0 for a sweep where every task was dropped, or one launched with
    # --n 0. Reaching a division by zero there loses the whole dashboard for a
    # run whose only honest content is "nothing completed".
    best_perf_pct = ((best_perf.get("passed", 0) / best_perf["n"]) * 100
                     if best_perf.get("n") else 0.0)
    best_value = min(sorted_rows, key=lambda x: x.get("cost_per_solved_usd", 999) if x.get("passed", 0) > 0 else 999) if sorted_rows else {}
    lowest_cost = min(sorted_rows, key=lambda x: x.get("total_as_run_usd", 999)) if sorted_rows else {}

    table_rows_html = ""
    chart_bars_html = ""

    for idx, r in enumerate(sorted_rows, 1):
        name = r.get("name", r.get("arm", f"Variant {idx}"))
        passed = r.get("passed", 0)
        n_val = r.get("n", 30)
        pr_pct = f"{(passed / n_val) * 100:.1f}%" if n_val > 0 else "0%"
        tot_usd = f"${r.get('total_as_run_usd', 0.0):.4f}"
        triage_usd = f"${r.get('total_triage_usd', 0.0):.4f}"
        cps = f"${r.get('cost_per_solved_usd', 0.0):.4f}" if passed > 0 else "N/A"
        avg_out = f"{r.get('avg_output_tokens', 0):.0f}"
        bar_w = (passed / n_val) * 100 if n_val > 0 else 0

        is_sj = "straitjacket" in name.lower() or "$0.00" in r.get("triage_mode", "")
        badge = '<span style="background:rgba(16,185,129,0.2);color:#10b981;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700;">STRAITJACKET</span>' if is_sj else '<span style="background:rgba(100,116,139,0.2);color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:0.75rem;">BASELINE</span>'

        table_rows_html += f"""
        <tr>
            <td style="text-align:center;font-weight:700;">{idx}</td>
            <td><strong>{name}</strong><br><span style="font-size:0.8rem;color:#94a3b8;">{r.get('models', '')}</span></td>
            <td style="text-align:center;">{badge}</td>
            <td style="text-align:center;font-weight:700;color:#60a5fa;">{passed}/{n_val} ({pr_pct})</td>
            <td style="text-align:center;font-family:monospace;">{tot_usd}</td>
            <td style="text-align:center;font-family:monospace;color:#10b981;">{triage_usd}</td>
            <td style="text-align:center;font-family:monospace;font-weight:700;color:#f59e0b;">{cps}</td>
            <td style="text-align:center;font-family:monospace;color:#94a3b8;">{avg_out}</td>
        </tr>
        """

        chart_bars_html += f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px;">
                <span style="color:#e2e8f0;font-weight:500;">{name}</span>
                <span style="color:#60a5fa;font-weight:700;">{pr_pct}</span>
            </div>
            <div style="background:#1e293b;height:8px;border-radius:4px;overflow:hidden;">
                <div style="background:linear-gradient(90deg,#3b82f6,#8b5cf6);height:100%;width:{bar_w}%;"></div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name} Multi-LLM Benchmark Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background: #070913; color: #f1f5f9; margin: 0; padding: 30px 20px 80px 20px; }}
        .container {{ max-width: 1240px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 40px; }}
        h1 {{ font-family: 'Outfit', sans-serif; font-size: 2.4rem; font-weight: 800; background: linear-gradient(90deg, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }}
        .subtitle {{ color: #94a3b8; font-size: 1.05rem; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 35px; }}
        .kpi-card {{ background: #0f1322; border: 1px solid #1e2540; border-radius: 14px; padding: 20px; border-top: 4px solid #3b82f6; }}
        .kpi-label {{ font-size: 0.8rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px; }}
        .kpi-val {{ font-family: 'Outfit', sans-serif; font-size: 1.8rem; font-weight: 700; color: #fff; }}
        .kpi-sub {{ font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }}
        .card {{ background: #0f1322; border: 1px solid #1e2540; border-radius: 14px; padding: 24px; margin-bottom: 30px; }}
        .card-title {{ font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 18px; border-left: 4px solid #3b82f6; padding-left: 10px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem; }}
        th {{ background: #1e293b; color: #e2e8f0; padding: 12px 16px; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
        td {{ padding: 14px 16px; border-bottom: 1px solid #1e2540; }}
        tr:hover td {{ background: #171d33; }}
        footer {{ text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 50px; border-top: 1px solid #1e2540; padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{dataset_name} Multi-LLM Benchmark Dashboard</h1>
            <p class="subtitle">Empirical Total Cost of Ownership (TCO) & Straitjacket Context Containment Evaluation</p>
        </header>

        <div class="kpi-grid">
            <div class="kpi-card" style="border-top-color:#3b82f6;">
                <div class="kpi-label">Top Performer</div>
                <div class="kpi-val">{best_perf_pct:.1f}%</div>
                <div class="kpi-sub">{best_perf.get('name', 'N/A')}</div>
            </div>
            <div class="kpi-card" style="border-top-color:#10b981;">
                <div class="kpi-label">Best Value ($/Solved)</div>
                <div class="kpi-val">${best_value.get('cost_per_solved_usd',0):.4f}</div>
                <div class="kpi-sub">{best_value.get('name', 'N/A')}</div>
            </div>
            <div class="kpi-card" style="border-top-color:#f59e0b;">
                <div class="kpi-label">Lowest Total Cost</div>
                <div class="kpi-val">${lowest_cost.get('total_as_run_usd',0):.4f}</div>
                <div class="kpi-sub">{lowest_cost.get('name', 'N/A')}</div>
            </div>
            <div class="kpi-card" style="border-top-color:#8b5cf6;">
                <div class="kpi-label">Evaluation Scope</div>
                <div class="kpi-val">{len(summary_rows)} Variants</div>
                <div class="kpi-sub">Evaluated on {summary_rows[0].get('n', 30) if summary_rows else 30} Tasks</div>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:2fr 1fr;gap:24px;">
            <div class="card">
                <div class="card-title">Comparative Architecture Rankings</div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead>
                            <tr>
                                <th style="width:40px;">#</th>
                                <th>Configuration</th>
                                <th style="text-align:center;">Type</th>
                                <th style="text-align:center;">Pass Rate</th>
                                <th style="text-align:center;">Total Cost</th>
                                <th style="text-align:center;">Triage USD</th>
                                <th style="text-align:center;">$/Solved</th>
                                <th style="text-align:center;">Avg Tokens</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows_html}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Pass Rate Comparison</div>
                {chart_bars_html}
            </div>
        </div>

        <footer>
            <p>Generated automatically by Straitjacket Multi-LLM Benchmark Suite &middot; 2026</p>
        </footer>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated HTML dashboard: {output_path}", flush=True)
    return output_path
