# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
The per-arm execution loop, factored out of `run_benchmark.py`.

Why this module exists
----------------------
`run_benchmark.py` owned the only copy of the loop that runs one architecture
over a task list: cache lookup, task-level retry, the discard-on-dispatch-error
rule, the containment/sub-task fields that must survive into the record, and the
summary row shape the reporter reads. Any single-arm runner written beside it
(see `run_classeval_opus5.py`) would have had to copy all of that, and a copy is
exactly how two arms end up being scored by two subtly different rules.

So the loop lives here and both entry points call it. A row produced by
`run_classeval_opus5.py` is then merge-compatible with a row produced by
`run_benchmark.py` by construction rather than by inspection.
"""

import json
import os
import time

from .client import DispatchError, reset_simulated_calls, simulated_calls, simulation_allowed
from .evaluator import (aggregate_containment as _aggregate_containment,
                        classeval_subtask_summary as _classeval_subtask_summary)


def load_cache(cache_file, no_cache=False):
    """The on-disk task cache, or an empty one when it is unusable/disabled."""
    if no_cache or not cache_file or not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as cf:
            return json.load(cf)
    except Exception:
        return {}


def save_cache(cache, cache_file):
    if not cache_file:
        return
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as cf:
        json.dump(cache, cf, indent=2)


def diagnostics_rollup(results):
    """Why this arm's tasks ended where they did, per attempt rather than per task.

    A pass rate answers "did the candidate resolve the row". It cannot answer
    "did the candidate ever get graded", and on a containerised dataset those
    are different questions with different owners: the first is about the
    models, the second about the harness. Reports 21 and 23 published the first
    without the second, and their 0-2% was read as a model result when ~89% of
    attempts had died before a test ran.

    `suite_reach_rate` is the fraction of *attempts* whose evidence came from
    the repository's own suite. `test_pass_ratio_avg` is the partial credit a
    binary verdict throws away.
    """
    reasons = {}
    attempts = reached = 0
    for r in results:
        per_attempt = r.get("guard_reasons")
        if per_attempt is None:
            continue
        for reason in per_attempt:
            attempts += 1
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                reached += 1
    ratios = [r.get("test_pass_ratio") for r in results
              if isinstance(r.get("test_pass_ratio"), (int, float))]
    routings = [r.get("routing") or {} for r in results]
    graded = [r for r in routings if r]
    return {
        "attempts": attempts,
        "suite_reached": reached,
        "suite_reach_rate": round(reached / attempts, 4) if attempts else None,
        # Sorted by count so the dominant failure is first in the report.
        "guard_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "test_pass_ratio_avg": (round(sum(ratios) / len(ratios), 4)
                                if ratios else None),
        "test_pass_ratio_n": len(ratios),
        # Routing provenance. `degraded` means the gate needed typed evidence
        # and did not get it, so the row did not test what the arm's name says.
        "frontier_used": sum(1 for t in graded if t.get("frontier_used")),
        "degraded": sum(1 for t in graded if t.get("degraded")),
        "routed": len(graded),
        "grounded": sum(1 for r in results
                        if (r.get("grounding") or {}).get("read")),
    }


def run_arm(cfg, problems, task_ids, cache=None, cache_file=None, no_cache=False,
            task_retries=1, sj_state=None, label=""):
    """Run one architecture variant over `task_ids` and return its summary row.

    `cache` is mutated in place and flushed to `cache_file` after every task, so
    an interrupted sweep keeps everything that completed.
    """
    v_id = cfg["id"]
    v_name = cfg["name"]
    fn = cfg["fn"]
    n = len(task_ids)

    print(f"{label}RUNNING: {v_name}")
    t0 = time.time()
    results = []
    failed_tasks = []
    passed_cnt = 0
    tot_usd = 0.0
    tot_out_tok = 0

    if cache is None:
        cache = {}
    if v_id not in cache:
        cache[v_id] = {}

    for t_idx, tid in enumerate(task_ids, start=1):
        prob = problems[tid]
        if not no_cache and tid in cache[v_id]:
            r = cache[v_id][tid]
            status_str = "PASS" if r.get("passed") else "FAIL"
            print(f"  [{t_idx}/{n}] {tid} ... [CACHED] {status_str} | "
                  f"cost=${r.get('as_run_usd', 0.0):.5f} | out_tok={r.get('output_tokens', 0)}",
                  flush=True)
        else:
            # A task whose API calls failed is not a result. Drop the partial
            # record and re-attempt it; only persist what completed.
            raw_r, dispatch_err = None, None
            for attempt in range(1, task_retries + 1):
                try:
                    reset_simulated_calls()
                    raw_r = fn(prob)
                    break
                except DispatchError as e:
                    dispatch_err = e
                    if attempt < task_retries:
                        wait = min(15 * attempt, 60)
                        print(f"  [{t_idx}/{n}] {tid} ... {e.kind.upper()} FAILURE "
                              f"(attempt {attempt}/{task_retries}); discarding the "
                              f"partial record, retrying in {wait}s", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"  [{t_idx}/{n}] {tid} ... GAVE UP after "
                              f"{task_retries} attempts: {e}", flush=True)

            if raw_r is None:
                # Recorded as an incomplete task, never as a pass or a fail,
                # and deliberately NOT written to the cache.
                failed_tasks.append({"task_id": tid, "kind": dispatch_err.kind,
                                     "error": str(dispatch_err)})
                continue

            r = {
                "task_id": tid,
                "passed": raw_r.get("passed", False),
                "as_run_usd": raw_r.get("as_run_usd", 0.0),
                "triage_usd": raw_r.get("triage_usd", 0.0),
                "output_tokens": raw_r.get("output_tokens", 0),
                "total_tokens": raw_r.get("total_tokens", 0),
                "repair_loops": raw_r.get("repair_loops", 0),
                # The containment ledger is a measurement, not a detail.
                # Provenance, so no later audit has to infer whether this
                # record came from a live call.
                "simulated_calls": simulated_calls(),
                "containment": raw_r.get("containment"),
                "retrievals": raw_r.get("retrievals"),
                "routing": raw_r.get("routing"),
                # Partial credit and *why* an attempt failed. Both are produced
                # per task and both used to stop here, at a whitelist that
                # never listed them -- so a dataset whose arms all score 0/50
                # published a table of zeroes while the record that would have
                # explained them (89% of attempts never reached a test) was
                # computed, dropped, and never written to disk.
                "test_pass_ratio": raw_r.get("test_pass_ratio"),
                "sbp": raw_r.get("sbp"),
                "guard_reason": raw_r.get("guard_reason", ""),
                "guard_reasons": raw_r.get("guard_reasons"),
                # The failing candidate itself, bounded. Without it a
                # `corrupt patch at line N` is unreadable after the fact.
                "candidate_patch": raw_r.get("candidate_patch"),
                "suite_reached": raw_r.get("suite_reached"),
                "attempts": raw_r.get("attempts"),
                "grounding": raw_r.get("grounding"),
                # ClassEval scores a task per method; the per-method records are
                # what makes a pass attributable to the model that wrote it.
                "subtasks": raw_r.get("subtasks"),
                "subtask_summary": raw_r.get("subtask_summary"),
                "error": str(raw_r.get("error", ""))[:500]
            }
            cache[v_id][tid] = r
            save_cache(cache, cache_file)
            status_str = "PASS" if r["passed"] else "FAIL"
            sub = r.get("subtask_summary") or {}
            sub_str = (f" | methods={sub.get('passed_subtasks', 0)}/{sub.get('n_subtasks', 0)}"
                       if sub.get("n_subtasks") else "")
            print(f"  [{t_idx}/{n}] {tid} ... {status_str} | cost=${r['as_run_usd']:.5f} | "
                  f"out_tok={r['output_tokens']}{sub_str}", flush=True)

        results.append(r)
        if r["passed"]:
            passed_cnt += 1
        tot_usd += r["as_run_usd"]
        tot_out_tok += r["output_tokens"]

    dt = time.time() - t0
    # Rates are over tasks that actually completed. Counting a dropped task
    # as a failure would silently understate the arm.
    n_done = len(results) or n
    pass_rate = (passed_cnt / n_done) if n_done > 0 else 0.0
    cost_per_solved = (tot_usd / passed_cnt) if passed_cnt > 0 else 0.0
    avg_out = tot_out_tok / n_done if n_done > 0 else 0.0

    # Triage USD: prefer what the arm actually spent; the per-repair
    # estimate is only a fallback for arms that do not report it.
    measured = sum(r.get("triage_usd", 0.0) for r in results)
    if measured > 0:
        triage_usd = round(measured, 6)
    elif "$0.00" in cfg.get("triage_mode", ""):
        triage_usd = 0.0000
    else:
        repairs = sum(r.get("repair_loops", 0) for r in results)
        triage_usd = round(repairs * 0.0018, 5)

    # Roll the per-method records up once per arm, so a report can show
    # pass rate BY TIER without re-reading every task record.
    sub_records = [sr for r in results for sr in (r.get("subtasks") or [])]
    subtask_rollup = _classeval_subtask_summary(sub_records) if sub_records else None

    simulated = sum(1 for t in results if t.get("simulated_calls"))
    diagnostics = diagnostics_rollup(results)
    if diagnostics["attempts"] and diagnostics["suite_reach_rate"] is not None \
            and diagnostics["suite_reach_rate"] < 0.5:
        print(f"  !! only {diagnostics['suite_reach_rate']:.0%} of attempts reached the "
              f"test suite; the rest died before grading "
              f"({', '.join(f'{k}={v}' for k, v in diagnostics['guard_reasons'].items())}). "
              "This arm's pass rate is a statement about the harness, not the models.",
              flush=True)
    if failed_tasks:
        print(f"  !! {len(failed_tasks)} task(s) never completed and are EXCLUDED "
              f"from this row: {', '.join(t['task_id'] for t in failed_tasks[:5])}"
              + (" ..." if len(failed_tasks) > 5 else ""), flush=True)

    summary = {
        "id": v_id,
        "name": v_name,
        "straitjacket": sj_state,
        # Provenance for the row: how many tasks completed, how many were
        # dropped, and whether any datapoint is simulated rather than live.
        "completed": len(results),
        "incomplete_tasks": failed_tasks,
        "simulated_tasks": simulated,
        "simulation_allowed": simulation_allowed(),
        "containment": _aggregate_containment(results),
        "diagnostics": diagnostics,
        # The exact task set this row was scored on. Rows in one report can
        # have different denominators (a dispatch failure drops a task), and a
        # pass rate over 40 tasks is not comparable with one over 50 unless
        # somebody checks. Recording the ids lets the reporter check.
        "task_ids": [r.get("task_id") for r in results],
        "subtask_rollup": subtask_rollup,
        "category": cfg.get("category", ""),
        "models": cfg.get("models", "N/A"),
        "triage_mode": cfg.get("triage_mode", "$0.00"),
        "n": n_done,
        "passed": passed_cnt,
        "pass_rate": round(pass_rate, 3),
        "total_as_run_usd": round(tot_usd, 6),
        "total_triage_usd": round(triage_usd, 6),
        "cost_per_solved_usd": round(cost_per_solved, 6),
        "avg_output_tokens": round(avg_out, 1),
        "seconds": round(dt, 1),
        "results": results
    }
    print(f"  -> Pass Rate: {passed_cnt}/{n} ({pass_rate:.1%}) | Cost: ${tot_usd:.4f} | "
          f"$/Solved: ${cost_per_solved:.4f}\n", flush=True)
    return summary


def print_scoreboard(summary_rows, dataset_name, n):
    """The final comparative table, identical in every entry point."""
    print("\n" + "=" * 95)
    print(f"FINAL COMPARATIVE TCO SCOREBOARD: {dataset_name.upper()} (N={n})")
    print("=" * 95)
    print(f"{'Configuration':<44} | {'Pass Rate':<10} | {'Total Cost ($)':<14} | "
          f"{'$/Solved':<10} | {'Triage USD'}")
    print("-" * 95)
    for s in summary_rows:
        pr_str = f"{s['passed']}/{s['n']} ({s['pass_rate']:.0%})"
        cps_str = f"${s['cost_per_solved_usd']:.4f}" if s['passed'] > 0 else "N/A"
        print(f"{s['name']:<44} | {pr_str:<10} | ${s['total_as_run_usd']:<13.4f} | "
              f"{cps_str:<10} | ${s['total_triage_usd']:.4f}")
    print("=" * 95 + "\n")
