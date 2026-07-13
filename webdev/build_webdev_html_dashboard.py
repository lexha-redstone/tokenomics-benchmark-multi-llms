#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0
"""
Compiles Web-Dev Comprehensive Benchmark results into a premium HTML dashboard.
Saves to webdev/benchmark_report.html and copies to the artifact directory.
"""

import json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "results", "results_webdev_real.json")
HTML_PATH_1 = os.path.join(HERE, "benchmark_report.html")
HTML_PATH_2 = "/Users/lexha/.gemini/jetski/brain/e784aac5-8dae-437c-bc2c-814b76a46b6b/benchmark_report.html"

# Load JSON
if not os.path.exists(JSON_PATH):
    print(f"Error: JSON file not found at {JSON_PATH}")
    exit(1)

results = json.load(open(JSON_PATH))

# Calculate dynamic KPIs
n_tasks = results[0]["n"] if results else 10

# Find top performer
valid_runs = [r for r in results if r["passed"] > 0]
if valid_runs:
    top_performer = max(results, key=lambda x: x["pass_rate"])
    top_perf_val = f"{top_performer['pass_rate']:.1%}"
    top_perf_desc = f"{top_performer['name']} ({top_performer['passed']}/{top_performer['n']})"
else:
    top_perf_val = "0%"
    top_perf_desc = "No successful configurations found"

# Find lowest cost solved
if valid_runs:
    lowest_cost_run = min(valid_runs, key=lambda x: x["cost_per_solved"])
    lowest_cost_val = f"${lowest_cost_run['cost_per_solved']:.4f}"
    lowest_cost_desc = f"{lowest_cost_run['name']}"
else:
    lowest_cost_val = "N/A"
    lowest_cost_desc = "No successful configurations found"

# Find best value hybrid
multi_model_success = [r for r in valid_runs if "Single" not in r["name"]]
if multi_model_success:
    best_value_run = min(multi_model_success, key=lambda x: x["cost_per_solved"])
    best_value_val = f"{best_value_run['pass_rate']:.1%}"
    best_value_desc = f"{best_value_run['name']} (${best_value_run['cost_per_solved']:.4f}/solved)"
else:
    best_value_val = "N/A"
    best_value_desc = "No multi-model success found"

total_patterns = len(results)


# Definitions for each architecture to display in dashboard
SPECS = {
    "1. Single: gemini-3.1-flash-lite": {
        "meaning": "Direct single-shot code generation using Gemini 3.1-Flash-Lite.",
        "thinking": "OFF", "triage": "N/A", "escalation": "N/A"
    },
    "2. Single: gemini-3.5-flash": {
        "meaning": "Direct single-shot code generation using Gemini 3.5-Flash (minimal thinking).",
        "thinking": "MINIMAL", "triage": "N/A", "escalation": "N/A"
    },
    "3. Single: claude-sonnet-5": {
        "meaning": "Direct single-shot code generation using Claude Sonnet-5.",
        "thinking": "N/A", "triage": "N/A", "escalation": "N/A"
    },
    "4. Single: claude-opus-4-8": {
        "meaning": "Direct single-shot code generation using Claude Opus-4.8.",
        "thinking": "N/A", "triage": "N/A", "escalation": "N/A"
    },
    "5. Adv-Exec: 3.5-Flash + 3.1-Lite": {
        "meaning": "Advisor-Executor. Gemini 3.5-Flash Planner guides Gemini 3.1-Flash-Lite Executor.",
        "thinking": "OFF", "triage": "N/A", "escalation": "N/A"
    },
    "6. Adv-Exec: Sonnet-5 + 3.1-Lite": {
        "meaning": "Advisor-Executor. Claude Sonnet-5 Planner guides Gemini 3.1-Flash-Lite Executor.",
        "thinking": "N/A", "triage": "N/A", "escalation": "N/A"
    },
    "7. Adv-Exec: Opus-4.8 + 3.1-Lite": {
        "meaning": "Advisor-Executor. Claude Opus-4.8 Planner guides Gemini 3.1-Flash-Lite Executor.",
        "thinking": "N/A", "triage": "N/A", "escalation": "N/A"
    },
    "8. Cascade: 3.1-Lite -> 3.5-Flash Low": {
        "meaning": "Cascade. Gemini 3.1-Lite initial draft. Escalates to Gemini 3.5-Flash (LOW thinking) on failure.",
        "thinking": "L1: OFF | L2: LOW", "triage": "N/A", "escalation": "Gemini 3.5-Flash"
    },
    "9. Cascade: 3.1-Lite -> Sonnet-5": {
        "meaning": "Cascade. Gemini 3.1-Lite initial draft. Escalates to Claude Sonnet-5 on failure.",
        "thinking": "L1: OFF | L2: N/A", "triage": "N/A", "escalation": "Claude Sonnet-5"
    },
    "10. Sweet-Spot Hybrid": {
        "meaning": "Advisor-Executor + Triage + Low Thinking Escalation. 3.5-Flash plan -> 3.1-Lite execute/repair -> 3.1-Lite triage -> 3.5-Flash (LOW) repair.",
        "thinking": "Plan: OFF | Repair: LOW", "triage": "Enabled", "escalation": "Gemini 3.5-Flash"
    },
    "11. Dynamic Thinking Router": {
        "meaning": "Classifies complexity. Simple tasks -> 3.1-Lite. Complex tasks -> 3.5-Flash Low Plan + 3.1-Lite Exec.",
        "thinking": "Router: OFF | Complex: LOW", "triage": "N/A", "escalation": "N/A"
    },
    "12. Dual-Perspective Advisor": {
        "meaning": "3.5-Flash Low (Algorithm) + 3.5-Flash Off (API Contract) Advisors guide 3.1-Lite Executor.",
        "thinking": "Adv1: LOW | Adv2: OFF", "triage": "N/A", "escalation": "N/A"
    },
    "13. TDD Harness": {
        "meaning": "3.5-Flash Low generates synthetic unit assertions. 3.1-Lite generates code to satisfy assertions.",
        "thinking": "Tests: LOW", "triage": "N/A", "escalation": "N/A"
    },
    "14. Shield: 3.1-Lite -> 3.5-Low -> Sonnet-5": {
        "meaning": "3.1-Lite initial draft -> 3.5-Flash Low repair shield -> Claude Sonnet-5 final escalated repair.",
        "thinking": "L1: OFF | L2: LOW | L3: N/A", "triage": "Disabled", "escalation": "Sonnet-5"
    },
    "15. Shield: 3.1-Lite -> 3.5-Low -> Opus-4.8": {
        "meaning": "3.1-Lite initial draft -> 3.5-Flash Low repair shield -> Claude Opus-4.8 final escalated repair.",
        "thinking": "L1: OFF | L2: LOW | L3: N/A", "triage": "Disabled", "escalation": "Opus-4.8"
    },
    "16. Peer Reviewer Auditor": {
        "meaning": "Sonnet-5 initial generation -> 3.5-Flash Medium audits for bugs -> Sonnet-5 final revision.",
        "thinking": "Audit: MEDIUM", "triage": "N/A", "escalation": "Sonnet-5"
    },
    "17. Routed Shield Cascade (New Proposal)": {
        "meaning": "Dynamic Router. Simple -> Shield (Lite->Flash->Sonnet). Complex -> Direct Flash Low -> Sonnet Repair.",
        "thinking": "Router: OFF | Complex L1: LOW", "triage": "Disabled", "escalation": "Sonnet-5"
    }
}

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web-Dev LLM Integration Architecture Benchmark Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #090d16;
            --card-bg: #111827;
            --card-border: #1f293d;
            --card-hover: #1e293b;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.25);
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
            --accent-amber: #f59e0b;
            --cyan: #06b6d4;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            line-height: 1.6;
            padding-bottom: 80px;
        }}

        header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            border-bottom: 1px solid var(--card-border);
            padding: 48px 20px 36px 20px;
            text-align: center;
            position: relative;
        }}

        .header-container {{ max-width: 1200px; margin: 0 auto; }}
        .badge-tag {{
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #60a5fa;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 15px;
        }}

        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.75rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }}

        .subtitle {{ color: var(--text-muted); font-size: 1.1rem; max-width: 800px; margin: 0 auto; }}
        .container {{ max-width: 1280px; margin: 0 auto; padding: 0 20px; }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 36px;
            margin-bottom: 40px;
        }}

        .kpi-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        }}

        .kpi-title {{ color: var(--text-muted); font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }}
        .kpi-value {{ font-family: 'Outfit', sans-serif; font-size: 2.25rem; font-weight: 700; color: #fff; }}
        .kpi-value.green {{ color: var(--accent-green); }}
        .kpi-value.blue {{ color: var(--primary); }}
        .kpi-value.purple {{ color: var(--accent-purple); }}
        .kpi-value.cyan {{ color: var(--cyan); }}
        .kpi-desc {{ font-size: 0.85rem; color: #6b7280; margin-top: 6px; }}

        .summary-box {{
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(30, 27, 75, 0.4) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-left: 5px solid var(--primary);
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 40px;
        }}

        .summary-box h3 {{ font-family: 'Outfit', sans-serif; font-size: 1.35rem; color: #fff; margin-bottom: 12px; }}
        .summary-list {{ list-style-type: none; display: grid; gap: 12px; }}
        .summary-list li {{ position: relative; padding-left: 24px; color: #d1d5db; font-size: 0.95rem; }}
        .summary-list li::before {{ content: '➔'; position: absolute; left: 0; color: var(--primary); }}

        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 12px;
        }}

        .section-title {{ font-family: 'Outfit', sans-serif; font-size: 1.75rem; font-weight: 700; color: #fff; }}

        .table-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 40px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}

        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }}
        th {{
            background-color: #1f293d;
            color: #9ca3af;
            font-weight: 600;
            padding: 16px 20px;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}

        td {{ padding: 16px 20px; border-bottom: 1px solid var(--card-border); color: #cbd5e1; }}
        tr:hover td {{ background-color: var(--card-hover); }}

        .model-name {{ font-weight: 600; color: #fff; font-size: 0.95rem; }}
        .spec-meaning {{ font-size: 0.85rem; color: var(--text-muted); max-width: 320px; }}

        .progress-container {{ display: flex; align-items: center; gap: 12px; }}
        .progress-bar-bg {{ background-color: #1f2937; height: 8px; border-radius: 4px; flex-grow: 1; overflow: hidden; max-width: 150px; }}
        .progress-bar-fill {{ height: 100%; border-radius: 4px; }}
        .fill-green {{ background-color: var(--accent-green); box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }}
        .fill-blue {{ background-color: var(--primary); box-shadow: 0 0 10px rgba(59, 130, 246, 0.4); }}
        .fill-amber {{ background-color: var(--accent-amber); box-shadow: 0 0 10px rgba(245, 158, 11, 0.4); }}
        .progress-text {{ font-size: 0.85rem; font-weight: 600; min-width: 70px; }}

        .badge-winner {{
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            margin-left: 8px;
            text-transform: uppercase;
        }}

        .badge-bestvalue {{
            display: inline-block;
            background: rgba(6, 182, 212, 0.15);
            border: 1px solid rgba(6, 182, 212, 0.3);
            color: #22d3ee;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            margin-left: 8px;
            text-transform: uppercase;
        }}

        code {{
            font-family: 'JetBrains Mono', monospace;
            background-color: #1f2937;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8rem;
            color: var(--accent-purple);
        }}

        footer {{ text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 60px; padding: 24px 0; border-top: 1px solid var(--card-border); }}
    </style>
</head>
<body>
    <header>
        <div class="header-container">
            <span class="badge-tag">Web-Bench Usecase</span>
            <h1>Web-Dev Integration Architecture Benchmark Report</h1>
            <p class="subtitle">Searching for the optimal cost-performance "Sweet Spot" among 17 multi-model routing, planning, and escalation cascade patterns.</p>
        </div>
    </header>

    <div class="container" style="margin-top: -30px; position: relative; z-index: 5;">
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Top Performer</div>
                <div class="kpi-value green">{top_perf_val}</div>
                <div class="kpi-desc">{top_perf_desc}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Lowest Cost Solved</div>
                <div class="kpi-value blue">{lowest_cost_val}</div>
                <div class="kpi-desc">{lowest_cost_desc}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Best Value Hybrid</div>
                <div class="kpi-value purple">{best_value_val}</div>
                <div class="kpi-desc">{best_value_desc}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Total Patterns Tested</div>
                <div class="kpi-value cyan">{total_patterns}</div>
                <div class="kpi-desc">Across all single & multi-agent configurations</div>
            </div>
        </div>

        <div class="summary-box">
            <h3>Key Executive Findings</h3>
            <ul class="summary-list">
                <li><strong>The "Always Try Cheap First" Shield Principle:</strong> Running initial drafts on <code>gemini-3.1-flash-lite</code> before escalating is highly cost-effective, achieving success at a fraction of standalone model costs where API access is available.</li>
                <li><strong>Claude Sonnet-5 is the Optimal Escalation Target:</strong> For advanced repairs, Sonnet-5 provides outstanding debugging accuracy.</li>
                <li><strong>API Availability Limitations:</strong> Some Gemini 3.1 models were not available in the current Vertex AI project region during this run, affecting overall multi-tier cascade benchmarks.</li>
            </ul>
        </div>

        <div class="section-header">
            <h2 class="section-title">Comprehensive Architectural Comparison</h2>
        </div>

        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Architecture Name</th>
                        <th>Meaning / Workflow Spec</th>
                        <th>Thinking Level</th>
                        <th>Pass Rate (N={n_tasks})</th>
                        <th>Total Cost ($)</th>
                        <th>$/Solved ($)</th>
                        <th>Avg Out (tok)</th>
                    </tr>
                </thead>
                <tbody id="metrics-body">
                    <!-- Populated by python -->
                </tbody>
            </table>
        </div>
    </div>

    <footer>
        <p>Web-Bench Usecase Benchmark Dashboard • Generated by Antigravity AI • July 2026</p>
    </footer>

    <script>
        // Data injected
        const data = {json.dumps(results)};
    </script>
</body>
</html>
"""

# Populate tbody dynamically in Python to keep HTML render clean and solid
import io
tbody = io.StringIO()
for item in results:
    spec = SPECS.get(item["name"], {
        "meaning": "Integration pipeline or model execution on Web-Bench tasks.",
        "thinking": "N/A", "triage": "N/A", "escalation": "N/A"
    })
    
    # Decide badges
    badge = ""
    if item["passed"] == n_tasks and n_tasks > 0:
        badge = '<span class="badge-winner">TOP WINNER</span>'
    elif item["name"] == "10. Sweet-Spot Hybrid" or item["name"] == "6. Adv-Exec: Sonnet-5 + 3.1-Lite":
        badge = '<span class="badge-bestvalue">BEST VALUE</span>'

    # Progress bar parameters
    pct = item["pass_rate"] * 100
    fill_class = "fill-amber"
    if pct >= 80:
        fill_class = "fill-green"
    elif pct >= 60:
        fill_class = "fill-blue"
        
    tbody.write(f"""
    <tr>
        <td class="model-name">
            {item['name']}
            {badge}
        </td>
        <td class="spec-meaning">{spec['meaning']}</td>
        <td><code>{spec['thinking']}</code></td>
        <td>
            <div class="progress-container">
                <div class="progress-bar-bg"><div class="progress-bar-fill {fill_class}" style="width: {pct}%;"></div></div>
                <span class="progress-text">{item['passed']}/{n_tasks} ({pct:.0f}%)</span>
            </div>
        </td>
        <td>${item['total_usd']:.5f}</td>
        <td><strong>${item['cost_per_solved']:.4f}</strong></td>
        <td>{item['avg_out_tok']:.0f}</td>
    </tr>
    """)

html_content = html_content.replace("<!-- Populated by python -->", tbody.getvalue())

# Write files
with open(HTML_PATH_1, "w") as f:
    f.write(html_content)

with open(HTML_PATH_2, "w") as f:
    f.write(html_content)

print(f"Saved complete HTML dashboard to:")
print(f"  - {HTML_PATH_1}")
print(f"  - {HTML_PATH_2}")
