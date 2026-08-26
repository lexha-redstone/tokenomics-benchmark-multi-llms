#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup fonts & style matching the repo standard
plt.rcParams['font.sans-serif'] = ['Apple SD Gothic Neo', 'Nanum Gothic', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(HERE, "swebench_pro_n20_scatter_plot.png")

# Data from reports/30_swebench-pro_straitjacket_n20.md
data = [
    {
        "id": "S0b",
        "title": "S0b: Claude Sonnet-5 Solo (3 rungs)",
        "models": "Claude Sonnet-5 x3",
        "cat": "Claude Solo Baseline",
        "color": "#ea580c", # deep orange
        "marker": "o",
        "size": 170,
        "cost": 3.3972,
        "passed": 6,
        "n": 20,
        "pass_rate": 30.0,
        "cost_per_solved": 0.5662,
        "avg_tokens": 5782.7,
        "suite_reached": "28/50 (56%)",
        "xytext": (25, 45),
        "box_color": "#ffedd5",
        "border_color": "#ea580c"
    },
    {
        "id": "S1",
        "title": "S1: Cascade (3.7-Flash → Sonnet-5, 3 rungs)",
        "models": "Gemini 3.7 Flash → Claude Sonnet-5",
        "cat": "Attempt-Count Cascade",
        "color": "#8b5cf6", # purple
        "marker": "o",
        "size": 160,
        "cost": 7.1195,
        "passed": 6,
        "n": 20,
        "pass_rate": 30.0,
        "cost_per_solved": 1.1866,
        "avg_tokens": 12354.9,
        "suite_reached": "22/50 (44%)",
        "xytext": (-45, -50),
        "box_color": "#f5f3ff",
        "border_color": "#8b5cf6"
    },
    {
        "id": "S2",
        "title": "★ S2: Evidence Gate (Flash → Sonnet / Opus, 3 rungs)",
        "models": "Gemini 3.7 Flash → Sonnet-5 / Opus-5",
        "cat": "Recommended (Evidence-Gated Escalation)",
        "color": "#dc2626", # red star
        "marker": "*",
        "size": 400,
        "cost": 7.8168,
        "passed": 8,
        "n": 20,
        "pass_rate": 40.0,
        "cost_per_solved": 0.9771,
        "avg_tokens": 13620.4,
        "suite_reached": "22/53 (42%)",
        "xytext": (-110, 48),
        "box_color": "#fee2e2",
        "border_color": "#dc2626"
    }
]

# Initialize Figure
fig, ax = plt.subplots(figsize=(16, 10.2), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f8fafc")

# Grid
ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", zorder=0)

# Pareto Optimal Frontier: S0b ($3.40, 30%) -> S2 ($7.82, 40%)
pareto_x = [3.3972, 7.8168]
pareto_y = [30.0, 40.0]
ax.plot(pareto_x, pareto_y, linestyle="--", color="#2563eb", linewidth=2.4, alpha=0.85, label="Pareto Frontier (Cost vs Accuracy Efficiency)", zorder=1)

# Accuracy ceiling line (S2 is top at 40.0%)
ax.axhline(40.0, color="#dc2626", linestyle=":", alpha=0.6, linewidth=1.5, zorder=1)
ax.text(9.55, 40.4, "Top Accuracy: 40.0% (S2 Evidence Gate: 8/20)", ha="right", va="bottom", fontsize=9.5, color="#b91c1c", fontweight="bold", fontstyle="italic")

# Plot points & labels
for item in data:
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
    
    is_s2 = (item["id"] == "S2")
    label_text = (
        f"{item['title']}\n"
        f"Cost: USD {item['cost']:.2f} | Solved: {item['pass_rate']:.1f}% ({item['passed']}/{item['n']})\n"
        f"Unit Cost: USD {item['cost_per_solved']:.4f} / solved | Avg Tokens: {item['avg_tokens']:.0f}\n"
        f"Suite Reached: {item['suite_reached']}"
    )
    
    fontsize = 9.5 if is_s2 else 8.5
    fontweight = "bold" if is_s2 or item["id"] == "S0b" else "normal"
    box_pad = 0.55 if is_s2 else 0.4
    lw = 2.0 if is_s2 else 1.0
    
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
ax.set_xlim(1.8, 9.8)
ax.set_ylim(20.0, 48.0)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

ax.set_xlabel("Total Benchmark Cost (USD)", fontsize=13, fontweight="bold", labelpad=12, color="#0f172a")
ax.set_ylabel("Performance: Resolved Rate % (N=20)", fontsize=13, fontweight="bold", labelpad=12, color="#0f172a")

plt.suptitle("SWE-bench Pro (N=20): Multi-LLM Benchmark Cost vs. Performance Tradeoff", fontsize=16, fontweight="bold", y=0.97, color="#0f172a")
ax.set_title("Evidence-Gated Escalation (S2) leads with 40.0% Pass Rate (8/20) at USD 0.9771/solved; Sonnet-5 Solo (S0b) offers lowest entry cost at USD 0.5662/solved", fontsize=11, fontweight="medium", pad=12, color="#475569")

# Custom Legend
legend_elements = [
    plt.Line2D([0], [0], marker='*', color='w', label='★ Top Performance: Evidence-Gated Escalation (S2: 40.0% @ USD 7.82)', markerfacecolor='#dc2626', markersize=16, markeredgecolor='#1e293b'),
    plt.Line2D([0], [0], marker='o', color='w', label='Claude Solo Baseline (S0b: Sonnet-5 @ USD 3.40, USD 0.5662/solved)', markerfacecolor='#ea580c', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Attempt-Count Cascade (S1: 3.7-Flash → Sonnet-5 @ USD 7.12)', markerfacecolor='#8b5cf6', markersize=10),
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

# Diagnostic Insight Callout Box
callout_text = (
    "Guardrail Diagnostic Note (Report 30):\n"
    "• Most failures were pre-grading 'apply_failed' errors (S0b: 22, S1: 25, S2: 27).\n"
    "• S2 successfully reached test suites with higher partial pass ratio (0.867 vs 0.759).\n"
    "• S1 (blind attempt cascade) is Pareto-dominated: cost USD 7.12 for 30% vs Sonnet USD 3.40."
)
ax.text(
    2.1, 21.8,
    callout_text,
    fontsize=8.8,
    color="#334155",
    bbox=dict(boxstyle="round,pad=0.6", fc="#f1f5f9", ec="#94a3b8", lw=1.1, alpha=0.95),
    va="bottom", ha="left"
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_PNG, dpi=300)
print(f"Successfully generated: {OUTPUT_PNG}")
