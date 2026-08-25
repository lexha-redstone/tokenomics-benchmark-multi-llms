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
ROOT = os.path.dirname(HERE)
JSON_PATH = os.path.join(ROOT, "classeval", "results", "classeval_classeval_results.json")
OUTPUT_PNG = os.path.join(HERE, "classeval_n91_scatter_plot.png")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data.get("summary", [])

# Arm definitions and fine-tuned callout positions
configs = [
    {
        "id": "C0a",
        "match": "C0a",
        "title": "C0a: Gemini 3.5-Flash-Lite Solo",
        "cat": "Gemini Solo",
        "color": "#0ea5e9", # sky blue
        "marker": "o",
        "size": 150,
        "xytext": (30, -38),
        "box_color": "#f0f9ff",
        "border_color": "#0ea5e9"
    },
    {
        "id": "C3",
        "match": "C3.",
        "title": "C3: Per-Method Lite (Control)",
        "cat": "Method-Level Decomposition",
        "color": "#10b981", # emerald
        "marker": "o",
        "size": 150,
        "xytext": (-60, 36),
        "box_color": "#ecfdf5",
        "border_color": "#10b981"
    },
    {
        "id": "C2",
        "match": "C2.",
        "title": "★ C2: Plan & Execute (3.7 Plan + Lite Exec)",
        "cat": "Recommended (Plan & Exec)",
        "color": "#dc2626", # red star
        "marker": "*",
        "size": 380,
        "xytext": (-70, 44),
        "box_color": "#fee2e2",
        "border_color": "#dc2626"
    },
    {
        "id": "C0c",
        "match": "C0c",
        "title": "C0c: Claude Sonnet-5 Solo",
        "cat": "Claude Solo",
        "color": "#d97706", # amber
        "marker": "o",
        "size": 150,
        "xytext": (-55, -44),
        "box_color": "#fef3c7",
        "border_color": "#d97706"
    },
    {
        "id": "C4",
        "match": "C4.",
        "title": "C4: H1 Router (Lite / 3.7 Flash)",
        "cat": "Method-Level Decomposition",
        "color": "#8b5cf6", # purple
        "marker": "o",
        "size": 140,
        "xytext": (25, -48),
        "box_color": "#f5f3ff",
        "border_color": "#8b5cf6"
    },
    {
        "id": "C0b",
        "match": "C0b",
        "title": "C0b: Gemini 3.7-Flash Solo (Low)",
        "cat": "Gemini Solo",
        "color": "#0284c7", # deep sky blue
        "marker": "o",
        "size": 140,
        "xytext": (40, -28),
        "box_color": "#f0f9ff",
        "border_color": "#0284c7"
    },
    {
        "id": "C5",
        "match": "C5.",
        "title": "C5: H1 + Plan (3.7 Plan + Routed Exec)",
        "cat": "Method-Level Decomposition",
        "color": "#8b5cf6",
        "marker": "o",
        "size": 140,
        "xytext": (55, -44),
        "box_color": "#f5f3ff",
        "border_color": "#8b5cf6"
    },
    {
        "id": "C1",
        "match": "C1.",
        "title": "C1: Cascade (Lite → 3.7 Low → 3.7 Med)",
        "cat": "Gemini Multi-Tier Ladder",
        "color": "#059669", # dark emerald
        "marker": "o",
        "size": 150,
        "xytext": (-65, 42),
        "box_color": "#ecfdf5",
        "border_color": "#059669"
    },
    {
        "id": "C0d",
        "match": "C0d",
        "title": "C0d: Claude Opus-5 Solo (Ceiling)",
        "cat": "Claude Solo",
        "color": "#ea580c", # deep orange
        "marker": "o",
        "size": 160,
        "xytext": (-45, 48),
        "box_color": "#ffedd5",
        "border_color": "#ea580c"
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
pareto_points = [
    (0.4040, 61.53846),
    (1.3886, 72.52747),
    (1.8594, 76.92308),
    (2.7094, 80.21978),
    (3.7120, 87.91209)
]
pareto_x, pareto_y = zip(*sorted(pareto_points, key=lambda p: p[0]))
ax.plot(pareto_x, pareto_y, linestyle="--", color="#2563eb", linewidth=2.2, alpha=0.8, label="Pareto Frontier (Cost vs Accuracy Efficiency)", zorder=1)

# Accuracy ceiling line
ax.axhline(87.91209, color="#ea580c", linestyle=":", alpha=0.6, linewidth=1.5, zorder=1)
ax.text(3.95, 88.2, "Accuracy Ceiling: 87.9% (Opus-5 Solo: 80/91)", ha="right", va="bottom", fontsize=9.5, color="#c2410c", fontweight="bold", fontstyle="italic")

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
    
    is_c2 = (item["id"] == "C2")
    label_text = (
        f"{item['title']}\n"
        f"Cost: ${item['cost']:.2f} | Solved: {item['pass_rate']:.1f}% ({item['passed']}/{item['n']})\n"
        f"Unit Cost: ${item['cost_per_solved']:.4f} / solved"
    )
    
    fontsize = 9.5 if is_c2 else 8.2
    fontweight = "bold" if is_c2 or item["id"] in ["C0d", "C0a"] else "normal"
    box_pad = 0.55 if is_c2 else 0.4
    lw = 2.0 if is_c2 else 1.0
    
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
ax.set_xlim(0.1, 4.15)
ax.set_ylim(57.0, 94.0)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

ax.set_xlabel("Total Benchmark Cost (USD)", fontsize=13, fontweight="bold", labelpad=12, color="#0f172a")
ax.set_ylabel("Performance: Problem Solved % (N=91)", fontsize=13, fontweight="bold", labelpad=12, color="#0f172a")

plt.suptitle("ClassEval (N=91): Multi-LLM Benchmark Cost vs. Performance Tradeoff", fontsize=16, fontweight="bold", y=0.97, color="#0f172a")
ax.set_title("Plan & Execute (C2) matches 3.7-Flash Solo accuracy (76.9%) at 86% of the cost (USD 1.86 vs USD 2.16); Opus-5 sets the ceiling at 87.9%", fontsize=11, fontweight="medium", pad=12, color="#475569")

# Custom Legend
legend_elements = [
    plt.Line2D([0], [0], marker='*', color='w', label='★ Recommended Sweet Spot: Plan & Execute (C2: 76.9% @ $1.86)', markerfacecolor='#dc2626', markersize=16, markeredgecolor='#1e293b'),
    plt.Line2D([0], [0], marker='o', color='w', label='Claude Solo Baselines (Sonnet-5 / Opus-5)', markerfacecolor='#ea580c', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Gemini Solo (3.5 Lite / 3.7 Flash)', markerfacecolor='#0284c7', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Gemini Multi-Tier Ladder (C1 Cascade)', markerfacecolor='#059669', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Method-Level Decomposition & Routing (C3 / C4 / C5)', markerfacecolor='#8b5cf6', markersize=10),
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
