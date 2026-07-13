import json, glob, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH_1 = os.path.join(HERE, "benchmark_report.html")
HTML_PATH_2 = "/Users/lexha/.gemini/jetski/brain/13962785-7ad3-4650-9dd4-0cb9a31ef91b/benchmark_report.html"
REPORT_DIR = os.path.join(HERE, "report")
REPORT_FILE = os.path.join(REPORT_DIR, "benchmark_report_20260709.html")

SPECS = {
    "1. Single: 3.5-Flash (OFF)": {
        "meaning": "Direct single-shot code generation using Gemini 3.5-Flash with thinking disabled.",
        "retries": "0 (Single Attempt)",
        "thinking": "OFF (budget=0)",
        "triage": "N/A"
    },
    "2. Single: 3.5-Flash (LOW)": {
        "meaning": "Direct single-shot code generation using Gemini 3.5-Flash with LOW thinking level (~1,200 thought tokens).",
        "retries": "0 (Single Attempt)",
        "thinking": "LOW",
        "triage": "N/A"
    },
    "3. Read/Write: 3.5-Flash + 3.1-Lite": {
        "meaning": "Gemini 3.5-Flash acts as Advisor (writes ~150-word prose guidance). Gemini 3.1-Flash-Lite acts as Executor (writes code).",
        "retries": "0 (Single Attempt)",
        "thinking": "OFF (Advisor & Executor)",
        "triage": "N/A"
    },
    "4. Pure Gemini 3-Tier: 3.1-Lite -> 3.5-Flash (LOW) -> 3.5-Flash (OFF)": {
        "meaning": "Level 1: 3.1-Lite initial gen. On failure, Level 2: 3.5-Flash (LOW) repair. On failure, Level 3: 3.5-Flash (OFF) final repair.",
        "retries": "Max 2 Repairs (3 Attempts)",
        "thinking": "L1: OFF | L2: LOW | L3: OFF",
        "triage": "Disabled (raw stderr)"
    },
    "5. Escalation Shield: 3.1-Lite -> 3.5-Flash (LOW) -> Sonnet-5": {
        "meaning": "Level 1: 3.1-Lite initial gen. Level 2: 3.5-Flash (LOW) acts as cheap repair shield. Level 3: Sonnet-5 final escalation.",
        "retries": "Max 2 Repairs (3 Attempts)",
        "thinking": "L1: OFF | L2: LOW | L3: N/A (Claude)",
        "triage": "Disabled (raw stderr)"
    },
    "6. 3-Tier Frontier: 3.1-Lite -> 3.5-Flash (MINIMAL) -> Opus-4.8": {
        "meaning": "Level 1: 3.1-Lite initial gen. Level 2: 3.5-Flash (MINIMAL) repair. Level 3: Claude Opus-4.8 final frontier repair.",
        "retries": "Max 2 Repairs (3 Attempts)",
        "thinking": "L1: OFF | L2: MINIMAL | L3: N/A (Claude)",
        "triage": "Disabled (raw stderr)"
    },
    "7. Smart Repair: 3.5-Flash + 3.1-Lite + Triage + 3.5-Flash (LOW)": {
        "meaning": "3.5-Flash Advisor + 3.1-Lite Executor. On failure, 3.1-Lite triages stderr into 12-line digest before 3.5-Flash (LOW) repair.",
        "retries": "Max 1 Repair (2 Attempts)",
        "thinking": "Advisor: OFF | Repair: LOW",
        "triage": "Enabled (12-line Flash-Lite Digest)"
    },
    "Single: claude-sonnet-5": {
        "meaning": "Direct single-shot code generation using Claude Sonnet-5.",
        "retries": "0 (Single Attempt)",
        "thinking": "N/A (Claude)",
        "triage": "N/A"
    },
    "Single: claude-opus-4-8": {
        "meaning": "Direct single-shot code generation using Claude Opus-4.8.",
        "retries": "0 (Single Attempt)",
        "thinking": "N/A (Claude)",
        "triage": "N/A"
    },
    "Single: gemini-3.1-flash-lite": {
        "meaning": "Direct single-shot code generation using Gemini 3.1-Flash-Lite.",
        "retries": "0 (Single Attempt)",
        "thinking": "OFF (budget=0)",
        "triage": "N/A"
    },
    "Arch 1: Read/Write Task Router (Opus-4.8/3.1-Lite)": {
        "meaning": "Inspects prompt (>1200 chars or >=3 libs). Read-heavy -> Opus-4.8 Planner + 3.1-Lite Exec. Write-heavy -> Direct 3.1-Lite.",
        "retries": "0 (Single Attempt)",
        "thinking": "OFF",
        "triage": "N/A"
    },
    "Arch 3: 3-Tier Escalation (3.1-Lite -> Sonnet-5 -> Opus-4.8)": {
        "meaning": "Level 1: 3.1-Lite initial gen. Level 2: Claude Sonnet-5 repair. Level 3: Claude Opus-4.8 final repair.",
        "retries": "Max 2 Repairs (3 Attempts)",
        "thinking": "L1: OFF | L2: N/A | L3: N/A",
        "triage": "Disabled"
    },
    "Config 1B: Ultra-Budget Cascade (3.1-Lite -> 3.5-Flash MINIMAL)": {
        "meaning": "Level 1: 3.1-Lite initial gen. On failure, Level 2: 3.5-Flash (MINIMAL) repair.",
        "retries": "Max 1 Repair (2 Attempts)",
        "thinking": "L1: OFF | L2: MINIMAL",
        "triage": "Disabled"
    },
    "Config 1A: Best-Value Cascade (3.1-Lite -> Sonnet-5)": {
        "meaning": "Level 1: 3.1-Lite initial gen. On failure, Level 2: Sonnet-5 repair.",
        "retries": "Max 1 Repair (2 Attempts)",
        "thinking": "L1: OFF | L2: N/A",
        "triage": "Disabled"
    },
    "Pattern 6: Tiered Thinking Ramping (3.5-Flash OFF -> LOW -> HIGH)": {
        "meaning": "Single-model 3.5-Flash ramping thinking budget on failure: OFF -> LOW -> HIGH.",
        "retries": "Max 2 Repairs (3 Attempts)",
        "thinking": "L1: OFF | L2: LOW | L3: HIGH",
        "triage": "Disabled"
    },
}

def get_all_dump_items():
    rdir = os.path.join(HERE, "results")
    items = []
    seen = set()
    for f in sorted(glob.glob(rdir + "/*.json")):
        if "cache" in f: continue
        try:
            d = json.load(open(f))
            if isinstance(d, list):
                for i in d:
                    if isinstance(i, dict):
                        name = i.get("name", i.get("architecture"))
                        n = i.get("n", i.get("total_tasks", 10))
                        passed = i.get("passed", i.get("passed_tasks", 0))
                        cost = i.get("total_as_run_usd", i.get("total_cost_usd", 0.0))
                        cps = i.get("cost_per_solved_usd", (cost/passed if passed>0 else -1.0))
                        key = (name, n)
                        if key not in seen and name:
                            seen.add(key)
                            items.append({"name": name, "n": n, "passed": passed, "total_cost": cost, "cost_per_solved": cps})
            elif isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, dict) and ("passed" in v or "passed_tasks" in v):
                        name = f"gemini-3.5-flash (Thinking: {k.upper()})"
                        n = 10
                        passed = v.get("passed", v.get("passed_tasks", 0))
                        cost = v.get("total_as_run_usd", v.get("total_cost_usd", 0.0))
                        cps = cost / passed if passed > 0 else -1.0
                        key = (name, n)
                        if key not in seen:
                            seen.add(key)
                            items.append({"name": name, "n": n, "passed": passed, "total_cost": cost, "cost_per_solved": cps})
        except Exception:
            pass
    items.sort(key=lambda x: (x['n'] != 30, -(x['passed']/x['n'])))
    return items

dump_items = get_all_dump_items()

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BigCodeBench-Hard LLM Architecture Benchmark Report</title>
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

        .tabs {{
            display: flex;
            gap: 10px;
            background: #111827;
            padding: 6px;
            border-radius: 12px;
            border: 1px solid var(--card-border);
            width: fit-content;
        }}

        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 18px;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .tab-btn.active {{
            background-color: var(--primary);
            color: #fff;
            box-shadow: 0 2px 10px var(--primary-glow);
        }}

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
            font-size: 0.775rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 14px 18px;
            border-bottom: 1px solid var(--card-border);
        }}

        td {{ padding: 14px 18px; border-bottom: 1px solid var(--card-border); color: #e5e7eb; vertical-align: top; }}
        tr:hover td {{ background-color: var(--card-hover); }}

        .model-name {{ font-weight: 600; color: #fff; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
        .badge-winner {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }}
        .badge-n30 {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }}
        .badge-n10 {{ background: rgba(139, 92, 246, 0.2); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }}

        .progress-container {{ display: flex; align-items: center; gap: 10px; width: 160px; }}
        .progress-bar-bg {{ flex-grow: 1; height: 8px; background-color: #1f2937; border-radius: 4px; overflow: hidden; }}
        .progress-bar-fill {{ height: 100%; border-radius: 4px; }}
        .fill-green {{ background: linear-gradient(90deg, #10b981, #34d399); }}
        .fill-blue {{ background: linear-gradient(90deg, #3b82f6, #60a5fa); }}
        .fill-amber {{ background: linear-gradient(90deg, #f59e0b, #fbbf24); }}

        code {{
            font-family: 'JetBrains Mono', monospace;
            background-color: #1f293d;
            color: #93c5fd;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.825rem;
        }}

        .spec-meaning {{ color: #d1d5db; font-size: 0.875rem; line-height: 1.4; max-width: 450px; }}
        .spec-badge {{ display: inline-block; background: #1e293b; color: #94a3b8; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 500; }}

        footer {{ text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 60px; }}
    </style>
</head>
<body>

    <header>
        <div class="header-container">
            <span class="badge-tag">Benchmark Dashboard • BigCodeBench-Hard</span>
            <h1>Comprehensive LLM Architecture Search</h1>
            <p class="subtitle">Complete performance, cost, retry count, and thinking level specifications across 35+ LLM architectural configurations.</p>
        </div>
    </header>

    <div class="container">
        
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Top Scale Pass Rate</div>
                <div class="kpi-value green">73.3%</div>
                <div class="kpi-desc">22 / 30 Solved (Pure Gemini 3-Tier)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Lowest Cost / Solved</div>
                <div class="kpi-value blue">$0.0017</div>
                <div class="kpi-desc">gemini-3.1-flash-lite (30.0% Pass Rate)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Single-Provider Parity</div>
                <div class="kpi-value purple">100%</div>
                <div class="kpi-desc">Gemini matches Sonnet/Opus 22/30 score</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Read/Write Split Savings</div>
                <div class="kpi-value cyan">50%</div>
                <div class="kpi-desc">$0.0087/solved vs $0.0170 single-shot</div>
            </div>
        </div>

        <div class="summary-box">
            <h3>💡 Executive Summary & Key Technical Findings</h3>
            <ul class="summary-list">
                <li><strong>Single-Provider Gemini Parity (73.3% / 22 Solved)</strong>: On the 30-task scale test, <code>Pure Gemini 3-Tier (3.1-Lite ➔ 3.5-Flash LOW ➔ 3.5-Flash OFF)</code> matched the top score of multi-vendor pipelines featuring Claude Sonnet-5 and Opus-4.8 (22/30 solved for all three) at an identical cost ($0.0275/solved). Enterprises can deploy a 100% Google Cloud native pipeline without Anthropic API keys.</li>
                <li><strong>3-Tier Escalation Outperforms Standalone Opus-4.8 (+20% Accuracy)</strong>: Standalone <code>claude-opus-4-8</code> solved 16/30 (53.3%) for $0.65. Every 3-Tier Escalation pipeline solved 21-22 tasks (70-73.3%) at equal or lower cost, demonstrating that unittest-driven repair loops are superior to single-shot frontier reasoning.</li>
                <li><strong>The "Thinking Budget" Trap (Gemini 3.5 Flash)</strong>: Single-shot <code>thinking_level="LOW"</code> (~1,200 thought tokens) is optimal for 3.5-Flash. Increasing to <code>MEDIUM</code> or <code>HIGH</code> increased cost and latency by 7x without increasing single-shot pass rate. Spending tokens on unittest repair loops is 5x more effective than upfront thinking!</li>
                <li><strong>Read/Write Task Splitting Efficiency</strong>: Offloading Read-heavy planning to 3.5-Flash (~150 words guidance) and Write-heavy code generation to 3.1-Flash-Lite matched standalone 3.5-Flash accuracy (40.0% on 30 tasks) at <strong>half the cost ($0.0087 vs $0.0170 per solved task)</strong>.</li>
            </ul>
        </div>

        <div class="section-header">
            <h2 class="section-title">Benchmark Explorer</h2>
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('scale30')">30-Task Scale Test (All 15 N=30 Runs)</button>
                <button class="tab-btn" onclick="switchTab('deepdive')">Architectural Deep Dive & Specs (All)</button>
                <button class="tab-btn" onclick="switchTab('intensive10')">10-Task Deep Dive (N=10)</button>
            </div>
        </div>

        <!-- TAB 1: ALL 15 30-TASK SCALE BENCHMARKS -->
        <div id="tab-scale30" class="tab-content">
            <div class="table-card">
                <table>
                    <thead>
                        <tr>
                            <th>Architecture / Configuration</th>
                            <th>Scale Tag</th>
                            <th>Pass Rate (30 Tasks)</th>
                            <th>Total Cost ($)</th>
                            <th>Cost / Solved ($)</th>
                        </tr>
                    </thead>
                    <tbody id="scale30-body">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- TAB 2: ARCHITECTURAL DEEP DIVE & PIPELINE SPECS (ALL RUNS) -->
        <div id="tab-deepdive" class="tab-content" style="display: none;">
            <div class="table-card">
                <table>
                    <thead>
                        <tr>
                            <th>Architecture Name</th>
                            <th>Scale Tag</th>
                            <th>Architecture Definition & Workflow Meaning</th>
                            <th>Retries / Attempts</th>
                            <th>Thinking Level Config</th>
                            <th>Pass Rate</th>
                            <th>$/Solved</th>
                        </tr>
                    </thead>
                    <tbody id="deepdive-body">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- TAB 3: 10-TASK INTENSIVE TEST (N=10) -->
        <div id="tab-intensive10" class="tab-content" style="display: none;">
            <div class="table-card">
                <table>
                    <thead>
                        <tr>
                            <th>Architecture / Pipeline (N=10)</th>
                            <th>Scale Tag</th>
                            <th>Pass Rate (10 Tasks)</th>
                            <th>Total Cost ($)</th>
                            <th>Cost / Solved ($)</th>
                        </tr>
                    </thead>
                    <tbody id="intensive-body">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>

    </div>

    <footer>
        <p>BigCodeBench-Hard Benchmark Dashboard • Generated by Antigravity AI • July 2026</p>
    </footer>

    <script>
        const SPECS = {json.dumps(SPECS)};
        const ALL_ITEMS = {json.dumps(dump_items)};

        function renderScale30() {{
            const body = document.getElementById('scale30-body');
            body.innerHTML = '';
            ALL_ITEMS.filter(i => i.n === 30).forEach(item => {{
                const pct = (item.passed / 30 * 100).toFixed(1);
                let fillClass = "fill-amber";
                if (item.passed >= 21) fillClass = "fill-green";
                else if (item.passed >= 16) fillClass = "fill-blue";

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="model-name">
                        ${{item.name}}
                        ${{item.passed >= 22 ? '<span class="badge-winner">TOP WINNER</span>' : (item.cost_per_solved > 0 && item.cost_per_solved < 0.01 ? '<span class="badge-winner">BEST VALUE</span>' : '')}}
                    </td>
                    <td><span class="badge-n30">N=30 Scale</span></td>
                    <td>
                        <div class="progress-container">
                            <div class="progress-bar-bg"><div class="progress-bar-fill ${{fillClass}}" style="width: ${{pct}}%;"></div></div>
                            <span class="progress-text">${{item.passed}}/30 (${{pct}}%)</span>
                        </div>
                    </td>
                    <td>$${{item.total_cost.toFixed(4)}}</td>
                    <td><strong>${{item.cost_per_solved > 0 ? '$' + item.cost_per_solved.toFixed(4) : 'N/A'}}</strong></td>
                `;
                body.appendChild(tr);
            }});
        }}

        function renderDeepDive() {{
            const body = document.getElementById('deepdive-body');
            body.innerHTML = '';
            ALL_ITEMS.forEach(item => {{
                const spec = SPECS[item.name] || {{
                    meaning: "Architectural pipeline evaluated on BigCodeBench-Hard dataset: " + item.name,
                    retries: item.name.includes("3-Tier") || item.name.includes("Cascade") || item.name.includes("Shield") ? "Max 2 Repairs (3 Attempts)" : "0 (Single Attempt)",
                    thinking: item.name.includes("LOW") ? "LOW" : (item.name.includes("MINIMAL") ? "MINIMAL" : "OFF"),
                    triage: item.name.includes("Triage") || item.name.includes("Smart") ? "Enabled" : "Disabled"
                }};

                const nBadge = item.n === 30 ? '<span class="badge-n30">N=30 Scale</span>' : '<span class="badge-n10">N=10 Deep Dive</span>';

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="model-name">
                        ${{item.name}}
                        ${{(item.passed / item.n) >= 0.70 ? '<span class="badge-winner">TOP SCORE</span>' : ''}}
                    </td>
                    <td>${{nBadge}}</td>
                    <td class="spec-meaning">${{spec.meaning}}</td>
                    <td><span class="spec-badge">${{spec.retries}}</span></td>
                    <td><code>${{spec.thinking}}</code></td>
                    <td><strong>${{item.passed}}/${{item.n}} (${{(item.passed/item.n * 100).toFixed(0)}}%)</strong></td>
                    <td>${{item.cost_per_solved > 0 ? '$' + item.cost_per_solved.toFixed(4) : 'N/A'}}</td>
                `;
                body.appendChild(tr);
            }});
        }}

        function renderIntensive() {{
            const body = document.getElementById('intensive-body');
            body.innerHTML = '';
            ALL_ITEMS.filter(i => i.n === 10).forEach(item => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="model-name">
                        ${{item.name}}
                        ${{item.passed >= 8 ? '<span class="badge-winner">TOP PASS</span>' : ''}}
                    </td>
                    <td><span class="badge-n10">N=10 Deep Dive</span></td>
                    <td>${{item.passed}}/10 (${{(item.passed*10).toFixed(0)}}%)</td>
                    <td>$${{item.total_cost.toFixed(4)}}</td>
                    <td><strong>${{item.cost_per_solved > 0 ? '$' + item.cost_per_solved.toFixed(4) : 'N/A'}}</strong></td>
                `;
                body.appendChild(tr);
            }});
        }}

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.style.display = 'none');
            
            if (tabId === 'scale30') {{
                document.querySelectorAll('.tab-btn')[0].classList.add('active');
                document.getElementById('tab-scale30').style.display = 'block';
            }} else if (tabId === 'deepdive') {{
                document.querySelectorAll('.tab-btn')[1].classList.add('active');
                document.getElementById('tab-deepdive').style.display = 'block';
            }} else {{
                document.querySelectorAll('.tab-btn')[2].classList.add('active');
                document.getElementById('tab-intensive10').style.display = 'block';
            }}
        }}

        window.onload = function() {{
            renderScale30();
            renderDeepDive();
            renderIntensive();
        }};
    </script>
</body>
</html>
"""

with open(HTML_PATH_1, "w") as f:
    f.write(html_content)

with open(HTML_PATH_2, "w") as f:
    f.write(html_content)

shutil.copy(HTML_PATH_1, REPORT_FILE)

print("Saved complete HTML dashboard to:")
print("  -", HTML_PATH_1)
print("  -", HTML_PATH_2)
print("  -", REPORT_FILE)
