#!/usr/bin/env python3
"""FeatureBench (N=48) cost-vs-performance, split by oracle budget.

Reads the two sweep result files that back reports/20 and reports/22.

Why two panels rather than one scatter. The eight arms did not run under one
protocol. `MAX_ORACLE_CALLS` was 3 when F0a/F0b/F1/F2 were executed and 2 when
F3/F4/F5/F6 were, which is visible in the records themselves: the first four
reach three rungs, the last four never exceed two. That is not a cosmetic
difference — with a two-entry `TIERS`, an arm capped at two oracle calls can
never reach the frontier rung at all, so `claude-opus-5` appears on 41 of F1's
tasks and on none of F4/F5/F6's. Plotting all eight against one cost axis would
compare a three-attempt ladder that bought Opus against a two-attempt ladder
that structurally could not, which is the mistake the repository's own README
warns about. Each panel is therefore internally comparable; across panels, only
the budget-matched pass rate in each label is.
"""
import os
import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.sans-serif'] = ['Apple SD Gothic Neo', 'Nanum Gothic', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULT_FILES = [
    os.path.join(ROOT, "featurebench", "results", "featurebench_featurebench_results.json"),
    os.path.join(ROOT, "featurebench", "results", "featurebench_all_results.json"),
]
OUTPUT_PNG = os.path.join(HERE, "featurebench_n48_scatter_plot.png")

summary = []
for path in RESULT_FILES:
    with open(path, "r", encoding="utf-8") as f:
        summary.extend(json.load(f).get("summary", []))
by_id = {row["id"]: row for row in summary}

APPLY_FAIL_MARKERS = ("patch did not apply", "contains no patch", "no `@@` hunk")


def wilson(k, n, z=1.96):
    """95% Wilson interval, in percent. At N=48 this is the whole story."""
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (centre - half), 100 * (centre + half)


def stats(row):
    """Application rate, budget-matched pass count, and observed rung depth."""
    applied = [r for r in row["results"]
               if not any(m in (r.get("error") or "") for m in APPLY_FAIL_MARKERS)]
    # Passes reached within two oracle calls, i.e. the first solve plus one
    # repair. This is the only quantity comparable across the two panels.
    matched = sum(1 for r in row["results"]
                  if r["passed"] and r.get("repair_loops", 0) <= 1)
    depth = max(len((r.get("routing") or {}).get("rungs") or []) for r in row["results"])
    frontier = sum(1 for r in row["results"]
                   if (r.get("routing") or {}).get("frontier_used"))
    return len(applied), sum(1 for r in applied if r["passed"]), matched, depth, frontier


PANELS = [
    {
        "key": "three",
        "title": "Executed at 3 oracle calls  (MAX_ORACLE_CALLS = 3)",
        "sub": "Reached the `claude-opus-5` rung — F1 on 41 tasks, F2 on 31",
        "xlim": (4.35, 10.95),
        "arms": [
            {"id": "fb_single_flash", "label": "F0a: Gemini 3.7-Flash Solo (low) ×3",
             "color": "#0284c7", "box": "#f0f9ff", "xytext": (96, 26)},
            {"id": "fb_single_sonnet", "label": "F0b: Claude Sonnet-5 Solo ×3",
             "color": "#ea580c", "box": "#ffedd5", "xytext": (-60, -52)},
            {"id": "fb_evidence_gate", "label": "F2: Evidence gate → Sonnet-5 / Opus-5",
             "color": "#8b5cf6", "box": "#f5f3ff", "xytext": (0, 58)},
            {"id": "fb_cascade", "label": "F1: Cascade → Sonnet-5 → Opus-5",
             "color": "#8b5cf6", "box": "#f5f3ff", "xytext": (-34, 52)},
        ],
    },
    {
        "key": "two",
        "title": "Executed at 2 oracle calls  (MAX_ORACLE_CALLS = 2)",
        "sub": "Frontier rung structurally unreachable — `claude-opus-5` used on 0 tasks",
        "xlim": (3.35, 9.55),
        "arms": [
            {"id": "fb_diff_aware_gate", "label": "F5: Diff-aware evidence gate",
             "color": "#dc2626", "box": "#fee2e2", "xytext": (30, 60), "star": True},
            {"id": "fb_diff_contract", "label": "F4: Diff-contract (strict unified diff)",
             "color": "#059669", "box": "#ecfdf5", "xytext": (2, -58)},
            {"id": "fb_spec_deconstruct", "label": "F6: Spec deconstruct (manifest)",
             "color": "#059669", "box": "#ecfdf5", "xytext": (104, 22)},
            {"id": "fb_plan_exec", "label": "F3: Opus-5 plans → Flash implements",
             "color": "#d97706", "box": "#fef3c7", "xytext": (-14, 58)},
        ],
    },
]

for panel in PANELS:
    for arm in panel["arms"]:
        row = by_id[arm["id"]]
        n_applied, applied_pass, matched, depth, frontier = stats(row)
        arm.update(
            passed=row["passed"], n=row["n"], cost=row["total_as_run_usd"],
            unit=row["cost_per_solved_usd"], rate=row["passed"] / row["n"] * 100.0,
            ci=wilson(row["passed"], row["n"]), applied=n_applied,
            applied_pass=applied_pass, matched=matched, depth=depth, frontier=frontier)

ALL = [a for p in PANELS for a in p["arms"]]
band_lo = max(a["ci"][0] for a in ALL)
band_hi = min(a["ci"][1] for a in ALL)

fig, axes = plt.subplots(
    1, 2, figsize=(19, 10.0), dpi=300, sharey=True,
    gridspec_kw={"width_ratios": [p["xlim"][1] - p["xlim"][0] for p in PANELS],
                 "wspace": 0.06})
fig.patch.set_facecolor("#ffffff")

for ax, panel in zip(axes, PANELS):
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", zorder=0)
    ax.axhspan(band_lo, band_hi, color="#64748b", alpha=0.11, zorder=0.5)

    ordered = sorted(panel["arms"], key=lambda a: a["cost"])
    frontier_pts, best = [], -1.0
    for a in ordered:
        if a["rate"] > best:
            frontier_pts.append(a)
            best = a["rate"]
    ax.plot([a["cost"] for a in frontier_pts], [a["rate"] for a in frontier_pts],
            linestyle="--", color="#2563eb", linewidth=2.2, alpha=0.8, zorder=1)

    for a in panel["arms"]:
        star = a.get("star", False)
        ax.scatter(a["cost"], a["rate"], color=a["color"], s=400 if star else 150,
                   marker="*" if star else "o",
                   edgecolors="#1e293b" if star else "white",
                   linewidths=1.8, alpha=0.95, zorder=4)
        text = (
            f"{a['label']}\n"
            f"Cost: \\${a['cost']:.2f} | Solved: {a['rate']:.1f}% ({a['passed']}/{a['n']})"
            f" | \\${a['unit']:.2f}/solved\n"
            f"Patch applied: {a['applied']}/{a['n']}"
            f" ({100 * a['applied'] / a['n']:.0f}%) → {a['applied_pass']}/{a['applied']} passed\n"
            f"Budget-matched (≤2 oracle calls): {a['matched']}/{a['n']}"
        )
        ax.annotate(
            text, xy=(a["cost"], a["rate"]), xytext=a["xytext"],
            textcoords="offset points",
            ha="center" if abs(a["xytext"][0]) < 30 else ("left" if a["xytext"][0] > 0 else "right"),
            va="center", fontsize=8.6, fontweight="bold" if star else "normal",
            bbox=dict(boxstyle=f"round,pad={0.5 if star else 0.4}", fc=a["box"],
                      ec=a["color"], lw=2.0 if star else 1.0, alpha=0.94),
            arrowprops=dict(arrowstyle="->", color="#475569", lw=1.1, shrinkA=4, shrinkB=4),
            zorder=5)

    ax.set_xlim(*panel["xlim"])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
    ax.set_xlabel("Total Benchmark Cost (USD)", fontsize=12, fontweight="bold",
                  labelpad=10, color="#0f172a")
    ax.set_title(f"{panel['title']}\n{panel['sub']}", fontsize=11.5,
                 fontweight="bold", color="#0f172a", pad=10)

axes[0].set_ylim(0.0, 18.6)
axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
axes[0].set_ylabel("Performance: Problem Solved % (N=48)", fontsize=12.5,
                   fontweight="bold", labelpad=12, color="#0f172a")
axes[0].text(4.42, 18.2,
             f"Every arm's 95% CI (Wilson) contains {band_lo:.1f}-{band_hi:.1f}%:\n"
             "no pairwise accuracy gap in the sweep reaches p<0.05",
             ha="left", va="top", fontsize=9.0, color="#475569", fontstyle="italic")

axes[1].text(
    0.035, 0.955,
    "What actually gates this benchmark\n"
    "Only 6-23% of candidate patches survive `git apply`; 331 of 353\n"
    "failures never reached a test. Conditional on applying, arms pass\n"
    "33-83%. The repair turn is told only \"patch did not apply\" — the\n"
    "`git apply` output is collected and then dropped, so the digest\n"
    "types it `shallow` and no evidence gate can fire on it.",
    transform=axes[1].transAxes, ha="left", va="top", fontsize=9.3, color="#7f1d1d",
    bbox=dict(boxstyle="round,pad=0.6", fc="#fff1f2", ec="#f43f5e", lw=1.4, alpha=0.95),
    zorder=6)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Single-model baseline (Gemini / Claude)',
               markerfacecolor='#0284c7', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Multi-LLM escalation (attempt / evidence gate)',
               markerfacecolor='#8b5cf6', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Frontier-planner hybrid (Opus-5 plans first)',
               markerfacecolor='#d97706', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Diff / spec engineering (contract, manifest)',
               markerfacecolor='#059669', markersize=10),
    plt.Line2D([0], [0], marker='*', color='w',
               label='F5 — cheapest total spend, but its gate never fired (= F4 rerun)',
               markerfacecolor='#dc2626', markersize=16, markeredgecolor='#1e293b'),
    plt.Line2D([0], [0], linestyle="--", color="#2563eb", lw=2.2,
               label='Cost frontier within a panel (accuracy ordering is within noise)'),
]
axes[1].legend(handles=legend_elements, loc="lower right", frameon=True,
               facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0,
               framealpha=0.95, borderpad=0.8)

plt.suptitle("FeatureBench (N=48): Cost vs. Performance — and why these eight arms "
             "are two experiments, not one",
             fontsize=16, fontweight="bold", y=0.982, color="#0f172a")
fig.text(0.5, 0.945,
         "The oracle budget was halved between the two sweeps, so the panels are not "
         "comparable on cost. Only the budget-matched line in each label is: at ≤2 oracle "
         "calls the arms score 1-7 of 48, and every 95% CI overlaps.",
         ha="center", va="top", fontsize=10.5, color="#475569")

plt.tight_layout(rect=[0, 0, 1, 0.878])
plt.savefig(OUTPUT_PNG, dpi=300)
print(f"Successfully generated: {OUTPUT_PNG}")
