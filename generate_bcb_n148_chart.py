#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup fonts & style
plt.rcParams['font.sans-serif'] = ['Apple SD Gothic Neo', 'Nanum Gothic', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "bigCodeBench-hard", "results", "bcb_router_results.json")
OUTPUT_PNG = os.path.join(HERE, "bigcodebench_hard_n148_scatter_plot.png")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data.get("summary", [])

# Mapping for each configuration
configs = [
    {
        "id": "R0a",
        "match": "R0a",
        "title": "R0a: Claude Sonnet-5 Solo (x3)",
        "cat": "Claude Solo",
        "color": "#d97706", # amber
        "marker": "o",
        "size": 150,
        "xytext": (-40, -42),
        "box_color": "#fef3c7",
        "border_color": "#d97706"
    },
    {
        "id": "R0b",
        "match": "R0b",
        "title": "R0b: Claude Opus-5 Solo (x3)",
        "cat": "Claude Solo",
        "color": "#ea580c", # deep orange
        "marker": "o",
        "size": 160,
        "xytext": (0, 72),
        "box_color": "#ffedd5",
        "border_color": "#ea580c"
    },
    {
        "id": "R1",
        "match": "R1.",
        "title": "R1: Gemini 3.7-Flash Solo (Low)",
        "cat": "Gemini Solo / Thinking",
        "color": "#0284c7", # sky blue
        "marker": "o",
        "size": 140,
        "xytext": (35, -42),
        "box_color": "#f0f9ff",
        "border_color": "#0284c7"
    },
    {
        "id": "R2",
        "match": "R2.",
        "title": "R2: Gemini 3.7-Flash Solo (Med)",
        "cat": "Gemini Solo / Thinking",
        "color": "#0284c7",
        "marker": "o",
        "size": 140,
        "xytext": (25, -44),
        "box_color": "#f0f9ff",
        "border_color": "#0284c7"
    },
    {
        "id": "R4",
        "match": "R4.",
        "title": "R4: Gemini Ladder (Lite→Low→Med)",
        "cat": "Gemini Multi-Tier",
        "color": "#059669", # emerald green
        "marker": "o",
        "size": 150,
        "xytext": (-55, 38),
        "box_color": "#ecfdf5",
        "border_color": "#059669"
    },
    {
        "id": "R5",
        "match": "R5.",
        "title": "R5: Thinking Ladder (Low→Med→High)",
        "cat": "Gemini Solo / Thinking",
        "color": "#0284c7",
        "marker": "o",
        "size": 140,
        "xytext": (-35, 42),
        "box_color": "#f0f9ff",
        "border_color": "#0284c7"
    },
    {
        "id": "R6",
        "match": "R6.",
        "title": "R6: Gemini Ladder → Opus-5",
        "cat": "Multi-LLM Escalation",
        "color": "#8b5cf6", # purple
        "marker": "o",
        "size": 150,
        "xytext": (-115, 52),
        "box_color": "#f5f3ff",
        "border_color": "#8b5cf6"
    },
    {
        "id": "R7",
        "match": "R7.",
        "title": "R7: 3.7(Med) → Opus-5",
        "cat": "Multi-LLM Escalation",
        "color": "#8b5cf6",
        "marker": "o",
        "size": 140,
        "xytext": (-35, -44),
        "box_color": "#f5f3ff",
        "border_color": "#8b5cf6"
    },
    {
        "id": "R8",
        "match": "R8.",
        "title": "R8: 3.7(Low→Med) → Opus-5",
        "cat": "Multi-LLM Escalation",
        "color": "#8b5cf6",
        "marker": "o",
        "size": 150,
        "xytext": (115, 52),
        "box_color": "#f5f3ff",
        "border_color": "#8b5cf6"
    },
    {
        "id": "R9",
        "match": "R9.",
        "title": "★ R9: Evidence-Gated Escalation",
        "cat": "Recommended (Evidence-Gated)",
        "color": "#dc2626", # red
        "marker": "*",
        "size": 380,
        "xytext": (-60, 44),
        "box_color": "#fee2e2",
        "border_color": "#dc2626"
    },
    {
        "id": "R10",
        "match": "R10.",
        "title": "R10: Ladder → Opus-5 (Fresh Solve)",
        "cat": "Multi-LLM Escalation",
        "color": "#8b5cf6",
        "marker": "o",
        "size": 140,
        "xytext": (-45, -45),
        "box_color": "#f5f3ff",
        "border_color": "#8b5cf6"
    }
]

# Match with JSON data
plot_data = []
for cfg in configs:
    matched = None
    for row in summary:
        if cfg["match"] in row["name"]:
            matched = row
            break
    if matched:
        p = matched["passed"]
        n = matched["n"]
        cost = matched["total_as_run_usd"]
        cps = matched["cost_per_solved_usd"]
        pct = (p / n) * 100.0
        plot_data.append({
            **cfg,
            "passed": p,
            "n": n,
            "cost": cost,
            "cost_per_solved": cps,
            "pass_rate": pct
        })

# Initialize Figure
fig, ax = plt.subplots(figsize=(16, 10.2), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f8fafc")

# Grid
ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", zorder=0)

# Pareto Optimal Frontier
# Points: R0a ($2.86, 66.89%) -> R4 ($3.90, 72.97%) -> R9 ($4.24, 81.08%) -> R6 ($5.65, 84.46%)
pareto_x = [2.861872, 3.904320, 4.237424, 5.653270]
pareto_y = [66.89189, 72.97297, 81.08108, 84.45946]
ax.plot(pareto_x, pareto_y, linestyle="--", color="#2563eb", linewidth=2.2, alpha=0.8, label="Pareto Frontier (Cost vs Accuracy Efficiency)", zorder=1)

# Accuracy ceiling line
ax.axhline(84.45946, color="#ea580c", linestyle=":", alpha=0.6, linewidth=1.5, zorder=1)
ax.text(8.4, 84.65, "Accuracy Ceiling: 84.5% (125/148)", ha="right", va="bottom", fontsize=9.5, color="#c2410c", fontweight="bold", fontstyle="italic")

# Plot points & labels
for item in plot_data:
    ax.scatter(
        item["cost"],
        item["pass_rate"],
        color=item["color"],
        s=item["size"],
        marker=item["marker"],
        edgecolors="#1e293b" if item["marker"] == "*" else "white",
        linewidths=1.8,
        alpha=0.95,
        zorder=4
    )
    
    # Text formatting (avoid raw $ for mathtext)
    is_r9 = (item["id"] == "R9")
    label_text = (
        f"{item['title']}\n"
        f"Cost: ${item['cost']:.2f} | Solved: {item['pass_rate']:.1f}% ({item['passed']}/{item['n']})\n"
        f"Unit Cost: ${item['cost_per_solved']:.4f} / solved"
    )
    
    fontsize = 9.5 if is_r9 else 8.2
    fontweight = "bold" if is_r9 or item["id"] in ["R0a", "R0b"] else "normal"
    box_pad = 0.55 if is_r9 else 0.4
    lw = 2.0 if is_r9 else 1.0
    
    bbox_props = dict(
        boxstyle=f"round,pad={box_pad}",
        fc=item["box_color"],
        ec=item["border_color"],
        lw=lw,
        alpha=0.94
    )
    
    ax.annotate(
        label_text,
        xy=(item["cost"], item["pass_rate"]),
        xytext=item["xytext"],
        textcoords="offset points",
        ha="center" if abs(item["xytext"][0]) < 30 else ("left" if item["xytext"][0] > 0 else "right"),
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        bbox=bbox_props,
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.1, shrinkA=4, shrinkB=4),
        zorder=5
    )

# Axis configuration
ax.set_xlim(2.2, 8.55)
ax.set_ylim(61.5, 90.5)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

ax.set_xlabel("Total Benchmark Cost (USD)", fontsize=13, fontweight="bold", labelpad=12, color="#0f172a")
ax.set_ylabel("Performance: Problem Solved % (N=148)", fontsize=13, fontweight="bold", labelpad=12, color="#0f172a")

plt.suptitle("BigCodeBench-Hard (N=148): Multi-LLM Benchmark Cost vs. Performance Tradeoff", fontsize=16, fontweight="bold", y=0.97, color="#0f172a")
ax.set_title("Evidence-Gated Escalation (R9) delivers 96% of Frontier Accuracy (81.1% vs 84.5%) for 74% of Frontier Spend (USD 4.24 vs USD 5.70)", fontsize=11, fontweight="medium", pad=12, color="#475569")

# Custom Legend
legend_elements = [
    plt.Line2D([0], [0], marker='*', color='w', label='★ Recommended: Evidence-Gated Escalation (R9: 81.1% @ $4.24)', markerfacecolor='#dc2626', markersize=16, markeredgecolor='#1e293b'),
    plt.Line2D([0], [0], marker='o', color='w', label='Claude Solo Baselines (Sonnet-5 / Opus-5)', markerfacecolor='#ea580c', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Gemini Solo / Thinking Ladders', markerfacecolor='#0284c7', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Gemini Multi-Tier Ladder (No Frontier)', markerfacecolor='#059669', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Multi-LLM Escalation / Hybrid Routing', markerfacecolor='#8b5cf6', markersize=10),
    plt.Line2D([0], [0], linestyle="--", color="#2563eb", lw=2.2, label='Pareto Frontier (Optimal Tradeoff Line)'),
]

ax.legend(
    handles=legend_elements,
    loc="lower right",
    frameon=True,
    facecolor="#ffffff",
    edgecolor="#cbd5e1",
    fontsize=9.5,
    framealpha=0.95,
    borderpad=0.8
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_PNG, dpi=300)
print(f"Successfully generated: {OUTPUT_PNG}")
