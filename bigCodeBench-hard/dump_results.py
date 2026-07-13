import json, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))
rdir = os.path.join(HERE, "results")
for f in sorted(glob.glob(rdir + "/*.json")):
    if "cache" in f:
        continue
    print("===", os.path.basename(f), "===")
    try:
        d = json.load(open(f))
        if isinstance(d, list):
            for item in d:
                if isinstance(item, dict) and "name" in item:
                    print(f"  {item['name']:<62} | Pass: {item.get('passed')}/{item.get('n')} | Cost: ${item.get('total_as_run_usd'):.4f} | $/Solved: ${item.get('cost_per_solved_usd'):.4f}")
    except Exception as e:
        print("  Error:", e)
