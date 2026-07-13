import json, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))
rdir = os.path.join(HERE, "results")
all_items = []

for f in sorted(glob.glob(rdir + "/*.json")):
    if "cache" in f:
        continue
    print("\n====================================")
    print("FILE:", os.path.basename(f))
    print("====================================")
    d = json.load(open(f))
    if isinstance(d, list):
        for item in d:
            if not isinstance(item, dict):
                continue
            name = item.get("name", item.get("architecture", "N/A"))
            n = item.get("n", item.get("total_tasks", 10))
            passed = item.get("passed", item.get("passed_tasks", 0))
            cost = item.get("total_as_run_usd", item.get("total_cost_usd", 0.0))
            cps = item.get("cost_per_solved_usd", (cost / passed if passed > 0 else -1.0))
            out_tok = item.get("avg_output_tokens", 0.0)
            print(f"  {name:<64} | N={n:<2} | Pass: {passed}/{n} ({passed/n:.0%}) | Cost: ${cost:.4f} | $/Solved: ${cps:.4f}")
            all_items.append({
                "source": os.path.basename(f),
                "name": name,
                "n": n,
                "passed": passed,
                "total_cost": cost,
                "cost_per_solved": cps,
                "avg_out": out_tok
            })
    elif isinstance(d, dict):
        if "results" in d and isinstance(d["results"], dict):
            for k, v in d["results"].items():
                n = 10
                passed = v.get("passed", 0)
                cost = v.get("total_as_run_usd", 0.0)
                cps = cost / passed if passed > 0 else -1.0
                out_tok = v.get("avg_output_tokens", 0.0)
                print(f"  {k:<64} | N={n:<2} | Pass: {passed}/{n} ({passed/n:.0%}) | Cost: ${cost:.4f} | $/Solved: ${cps:.4f}")
                all_items.append({
                    "source": os.path.basename(f),
                    "name": k,
                    "n": n,
                    "passed": passed,
                    "total_cost": cost,
                    "cost_per_solved": cps,
                    "avg_out": out_tok
                })
        elif "results" in d and isinstance(d["results"], list):
            for item in d["results"]:
                name = item.get("name", item.get("architecture", "N/A"))
                n = item.get("n", item.get("total_tasks", 10))
                passed = item.get("passed", item.get("passed_tasks", 0))
                cost = item.get("total_as_run_usd", item.get("total_cost_usd", 0.0))
                cps = item.get("cost_per_solved_usd", (cost / passed if passed > 0 else -1.0))
                out_tok = item.get("avg_output_tokens", 0.0)
                print(f"  {name:<64} | N={n:<2} | Pass: {passed}/{n} ({passed/n:.0%}) | Cost: ${cost:.4f} | $/Solved: ${cps:.4f}")
                all_items.append({
                    "source": os.path.basename(f),
                    "name": name,
                    "n": n,
                    "passed": passed,
                    "total_cost": cost,
                    "cost_per_solved": cps,
                    "avg_out": out_tok
                })
        else:
            for k, v in d.items():
                if isinstance(v, dict) and ("passed" in v or "passed_tasks" in v):
                    name = f"gemini-3.5-flash (Thinking: {k.upper()})"
                    n = 10
                    passed = v.get("passed", v.get("passed_tasks", 0))
                    cost = v.get("total_as_run_usd", v.get("total_cost_usd", 0.0))
                    cps = cost / passed if passed > 0 else -1.0
                    out_tok = v.get("avg_output_tokens", 0.0)
                    print(f"  {name:<64} | N={n:<2} | Pass: {passed}/{n} ({passed/n:.0%}) | Cost: ${cost:.4f} | $/Solved: ${cps:.4f}")
                    all_items.append({
                        "source": os.path.basename(f),
                        "name": name,
                        "n": n,
                        "passed": passed,
                        "total_cost": cost,
                        "cost_per_solved": cps,
                        "avg_out": out_tok
                    })
