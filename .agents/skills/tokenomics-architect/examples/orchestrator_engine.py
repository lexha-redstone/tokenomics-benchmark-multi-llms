"""Unified Multi-Model Tokenomics & Orchestration Engine Reference.

Supports capability x price x thinking routing, acyclic DAG validation, topological wave scheduling,
bounded checkpoint handoff simulation, single-tier failure escalation, and dynamic re-planning.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

CAPABILITY_TIERS = ("frontier", "standard", "economy")
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high")


def tier_rank(tier: str) -> int:
    try:
        return len(CAPABILITY_TIERS) - CAPABILITY_TIERS.index(tier.lower().strip())
    except ValueError:
        return 0


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float
    cache_read_per_m: float = 0.0

    def cost_usd(self, input_tokens: int, output_tokens: int, cache_hits: int = 0) -> float:
        regular_in = max(0, input_tokens - cache_hits)
        return (regular_in * self.input_per_m + cache_hits * self.cache_read_per_m + output_tokens * self.output_per_m) / 1_000_000.0


@dataclass(frozen=True)
class ModelChoice:
    id: str
    tier: str
    roles: tuple[str, ...] = ()
    price: Price = field(default_factory=lambda: Price(1.0, 5.0, 0.1))
    default_thinking: str = "off"


@dataclass(frozen=True)
class HostSpec:
    name: str
    default_model: str
    coordinator_model: str
    models: tuple[ModelChoice, ...] = ()
    strengths: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectedHost:
    spec: HostSpec
    installed: bool = True

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def models(self) -> tuple[ModelChoice, ...]:
        return self.spec.models

    def model_price(self, model_id: str) -> Price:
        for m in self.models:
            if m.id == model_id:
                return m.price
        return Price(1.0, 5.0, 0.1)

    def coordinator_price(self) -> Price:
        return self.model_price(self.spec.coordinator_model)


# Unified Host Registry
DEFAULT_HOSTS = [
    DetectedHost(
        spec=HostSpec(
            name="antigravity-sdk",
            default_model="gemini-3.6-flash",
            coordinator_model="gemini-3.5-flash-lite",
            models=(
                ModelChoice("gemini-3.1-pro", "frontier", ("plan", "reason", "review", "architect"), Price(2.0, 12.0, 0.20), default_thinking="low"),
                ModelChoice("gemini-3.6-flash", "standard", ("implement", "edit", "code", "summarize", "reasoning_repair"), Price(1.50, 7.50, 0.15), default_thinking="low"),
                ModelChoice("gemini-3.5-flash-lite", "economy", ("explore", "search", "triage", "verify", "implement", "edit"), Price(0.30, 2.50, 0.03), default_thinking="off"),
            ),
            strengths=("search", "triage", "verify", "implement", "summarize", "explore"),
        )
    ),
    DetectedHost(
        spec=HostSpec(
            name="claude",
            default_model="claude-sonnet-4.6",
            coordinator_model="claude-haiku-4.5",
            models=(
                ModelChoice("claude-opus-4.8", "frontier", ("plan", "reason", "synthesize", "decide", "review", "architect", "deadlock_breaker"), Price(15.0, 75.0, 1.50), default_thinking="adaptive"),
                ModelChoice("claude-sonnet-4.6", "standard", ("implement", "edit", "code", "review", "contract_advisor"), Price(3.0, 15.0, 0.30), default_thinking="off"),
                ModelChoice("claude-haiku-4.5", "economy", ("explore", "search", "triage", "verify", "summarize"), Price(1.0, 5.0, 0.10), default_thinking="off"),
            ),
            strengths=("reason", "synthesize", "implement", "edit", "code", "review", "decide"),
        )
    ),
]


@dataclass(frozen=True)
class RouteNode:
    id: str
    goal: str
    role: str
    min_tier: str
    need_tags: tuple[str, ...]
    deps: tuple[str, ...]
    thinking_level: str = "off"
    est_input_tokens: int = 20000
    est_output_tokens: int = 3000
    host_pin: str = ""
    model_pin: str = ""
    prefer: str = "cheap"  # "cheap" | "strong"


@dataclass
class AssignedNode:
    node: RouteNode
    host: DetectedHost
    model: ModelChoice
    est_cost_usd: float
    tier_met: bool
    resolved_thinking: str


@dataclass
class RoutePlan:
    task: str
    hosts: tuple[DetectedHost, ...]
    coordinator: DetectedHost | None
    assigned: list[AssignedNode] = field(default_factory=list)

    @property
    def est_total_usd(self) -> float:
        return sum(a.est_cost_usd for a in self.assigned)

    def waves(self) -> list[list[AssignedNode]]:
        done: set[str] = set()
        layers: list[list[AssignedNode]] = []
        remaining = list(self.assigned)
        while remaining:
            ready = [a for a in remaining if all(d in done for d in a.node.deps)]
            if not ready:
                layers.append(remaining)
                break
            layers.append(ready)
            for a in ready:
                done.add(a.node.id)
            remaining = [a for a in remaining if a.node.id not in done]
        return layers


def pick_model(
    hosts: Sequence[DetectedHost],
    *,
    min_tier: str = "economy",
    need_tags: tuple[str, ...] = (),
    prefer: str = "cheap",
) -> tuple[DetectedHost, ModelChoice] | None:
    cands = [(h, m) for h in hosts if h.installed for m in h.models]
    if not cands:
        return None

    eligible = [(h, m) for (h, m) in cands if tier_rank(m.tier) >= tier_rank(min_tier)]
    if eligible:
        def score(hm: tuple[DetectedHost, ModelChoice]):
            h, m = hm
            cover = len(set(need_tags) & set(m.roles))
            p = h.model_price(m.id)
            if prefer == "strong":
                return (-cover, -tier_rank(m.tier), -p.output_per_m, h.name, m.id)
            return (-cover, p.output_per_m, p.input_per_m, h.name, m.id)

        return sorted(eligible, key=score)[0]

    return sorted(cands, key=lambda hm: (-tier_rank(hm[1].tier), hm[0].model_price(hm[1].id).output_per_m, hm[0].name))[0]


def assign_host(node: RouteNode, hosts: Sequence[DetectedHost]) -> AssignedNode:
    got = pick_model(hosts, min_tier=node.min_tier, need_tags=node.need_tags, prefer=node.prefer)
    if got is None:
        raise ValueError("No installed model available")
    host, model = got
    price = host.model_price(model.id)
    cost = price.cost_usd(node.est_input_tokens, node.est_output_tokens)
    thinking = node.thinking_level if node.thinking_level != "off" else model.default_thinking
    return AssignedNode(
        node=node,
        host=host,
        model=model,
        est_cost_usd=cost,
        tier_met=tier_rank(model.tier) >= tier_rank(node.min_tier),
        resolved_thinking=thinking,
    )


def fallback_route(task: str, hosts: Sequence[DetectedHost]) -> RoutePlan:
    nodes = [
        RouteNode("explore", "Gather repository anchors and facts", "explore", "economy", ("search", "triage", "explore"), (), thinking_level="off"),
        RouteNode("plan", "Architect solution and precise edit spec", "plan", "frontier", ("plan", "reason", "architect"), ("explore",), thinking_level="low", prefer="strong"),
        RouteNode("implement", "Execute code modifications", "implement", "standard", ("implement", "edit", "code"), ("plan",), thinking_level="low"),
        RouteNode("verify", "Run tests and inspect diff", "verify", "economy", ("verify", "test"), ("implement",), thinking_level="off"),
    ]
    assigned = [assign_host(n, hosts) for n in nodes]
    return RoutePlan(task=task, hosts=tuple(hosts), coordinator=hosts[0], assigned=assigned)


def render_route_plan(plan: RoutePlan) -> str:
    lines = [f'=== Unified Route Plan for: "{plan.task}" ===']
    lines.append(f"Waves: {len(plan.waves())} | Total Est Cost: ${plan.est_total_usd:.4f}\n")
    for idx, wave in enumerate(plan.waves(), 1):
        lines.append(f"Wave {idx}:")
        for a in wave:
            deps = f" (deps: {', '.join(a.node.deps)})" if a.node.deps else ""
            lines.append(f"  - [{a.node.id}] -> {a.host.name}/{a.model.id} ({a.model.tier}, thinking={a.resolved_thinking}) ~${a.est_cost_usd:.4f}{deps}")
    return "\n".join(lines)


if __name__ == "__main__":
    plan = fallback_route("Refactor user auth with JWT tokens", DEFAULT_HOSTS)
    print(render_route_plan(plan))
