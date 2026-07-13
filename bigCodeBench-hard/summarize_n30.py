#!/usr/bin/env python3
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(HERE, "results", "cache_gemini35_n30.json")
OUT_FILE = os.path.join(HERE, "results", "results_gemini35_sweetspot_n30.json")

if not os.path.exists(CACHE_FILE):
    print("No cache file found!")
    exit(1)

cache = json.load(open(CACHE_FILE))
summaries = []

for cfg_name, tasks in cache.items():
    task_results = list(tasks.values())
    n = len(task_results)
    if n == 0:
        continue
    passed_cnt = sum(1 for r in task_results if r.get("passed"))
    tot_cost = sum(r.get("as_run_usd", 0.0) for r in task_results)
    avg_out = sum(r.get("output_tokens", 0) for r in task_results) / n
    cps = (tot_cost / passed_cnt) if passed_cnt > 0 else -1.0

    summaries.append({
        "name": cfg_name,
        "n": n,
        "passed": passed_cnt,
        "pass_rate": round(passed_cnt / n, 3),
        "total_as_run_usd": round(tot_cost, 4),
        "cost_per_solved_usd": round(cps, 4) if cps >= 0 else -1.0,
        "avg_output_tokens": round(avg_out, 1)
    })

with open(OUT_FILE, "w") as f:
    json.dump(summaries, f, indent=2)

print("\n" + "=" * 98)
print("TOP GEMINI 3.5-FLASH ARCHITECTURES 30-TASK BENCHMARK (BigCodeBench-Hard, N=30)")
print("=" * 98)
print(f"{'Configuration':<64} | {'Pass Rate':<12} | {'Total Cost ($)':<12} | {'$/Solved':<10}")
print("-" * 98)
for s in summaries:
    cps_str = f"${s['cost_per_solved_usd']:.4f}" if s['cost_per_solved_usd'] >= 0 else "N/A"
    print(f"{s['name']:<64} | {s['passed']}/{s['n']} ({s['pass_rate']:.0%})  | ${s['total_as_run_usd']:<11.4f} | {cps_str:<10}")
print("=" * 98)
