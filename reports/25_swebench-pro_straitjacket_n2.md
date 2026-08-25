# Comparative TCO Report: `straitjacket` on `SWE-bench Pro` (N=2)

This report presents the empirical evaluation of multi-model collaboration architectures and `straitjacket` context containment on the **SWE-bench Pro** benchmark.

> **Harness provenance** — digests produced by `ctx-harness` v0.35.1 via the `library` backend (upstream `ctx.digest` profile registry, unmodified). Uncontained arms send at most 2500 chars of raw output (`SJ_RAW_CAP`).

## 1. Comparative TCO & Performance Table

| Configuration | Models | Evidence Treatment | Raw Pass Rate | Effective Pass Rate | Total Cost (USD) | Triage Cost (USD) | Cost / Solved Task ($/solved) | Avg Output Tokens |
|---|---|---|---|---|---|---|---|---|
| **S0b. Single: claude-sonnet-5 (3 rungs)** | `Claude Sonnet-5 x3` | Straitjacket contained digest ($0.00) | 1/2 (50.0%) | **1/2 (50.0%)** | `$0.2194` | `$0.0000` | **`$0.2194`** | `1791.5` |

---

## 2. Attempt Diagnostics

What happened to every *attempt*, not every task. `Suite reached` is the share of attempts whose evidence came from the repository's own test run; the rest died at a guard before grading and say nothing about the model. `Avg partial` is the mean `test_pass_ratio` over graded attempts — the credit a binary resolved/not-resolved verdict discards. `Frontier` and `Degraded` are the router's own record: how many tasks actually reached the frontier rung, and how many were routed by a gate that wanted typed evidence and did not get it.

| Configuration | Attempts | Suite reached | Avg partial | Grounded | Frontier used | Degraded | Dominant guard failure |
|---|---|---|---|---|---|---|---|
| **S0b. Single: claude-sonnet-5 (3 rungs)** | 4 | 2/4 (50%) | `0.500` (n=2) | 2 | 0 | 0 | `apply_failed` × 2 |

---

## 3. Context Containment Receipt

Measured by the harness itself for every captured run in the sweep. Every arm executes through the harness; `Captured` differs between them because they make different numbers of attempts and their candidate solutions print different amounts. What the comparison turns on is which payload each arm put in front of the model.

- **Captured** — everything the execution produced; the store holds all of it.
- **Sent to model** — what this arm actually placed in the repair prompt.
- **Native baseline** — what the *untreated* path would have forwarded for the same failures (the failing stream, tail-truncated).
- **Δ vs native** — the A/B advantage. This, not `Captured − Sent`, is what the treatment bought: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

| Configuration | Treatment | Profiles | Captures | Captured | Sent to model | Native baseline | Δ vs native |
|---|---|---|---|---|---|---|---|
| **S0b. Single: claude-sonnet-5 (3 rungs)** | straitjacket | `pytest/v1, text/v1` | 2 | `2,057` | **`1,004`** | `1,004` | `+0` (+0%) |

---

## 4. Key TCO & Architectural Insights

1. **Containment, not compression**: the straitjacket arms send the harness's own digest for the failing run — profile-detected, coverage-attested, and carrying `ctx get` / `ctx search` addresses for every omitted region. No triage model is called, so their triage cost is $0.0000.
2. **Residency, not just spend**: the containment table reports what each arm sent against what the untreated path would have sent. Dollars measure one turn; residency measures every turn those bytes would have stayed in the transcript. A negative delta is reported as readily as a positive one.
3. **Omission is not amnesia**: what the digest leaves out stays retrievable at an exact address, so a shorter prompt does not mean lost evidence.
4. **Where containment does nothing**: a run whose whole output is a handful of lines has nothing to contain, and its delta lands at or below zero. That is reported rather than hidden — short output is not automatically good output.
5. **Cost per solved task**: `S0b. Single: claude-sonnet-5 (3 rungs)` is the cheapest per solved task at `$0.2194`; `S0b. Single: claude-sonnet-5 (3 rungs)` has the highest pass rate at 50%.
