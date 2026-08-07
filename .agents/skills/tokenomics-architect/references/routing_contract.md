# Unified Routing Contract & `ctx.route/v1` Schema Specification

This document defines the strict specification for task decomposition, thinking budget allocation, and multi-model DAG routing.

---

## 1. Unified Coordinator Contract (`ROUTING_CONTRACT`)

The coordinator is a low-cost model (e.g., `gemini-3.5-flash-lite`, `gpt-5.4-nano`, or `claude-haiku-4.5`). It decomposes the user task into a bounded Directed Acyclic Graph (DAG) and assigns each node to a capability tier, role, and optimal thinking level.

```text
You are the COORDINATOR of a multi-harness tokenomics collaboration. Do NOT do the task yourself.
Split it into subtasks and assign each to the model/harness whose capability and thinking budget fit,
spending the cheapest resource that can do the work reliably. Output ONLY a JSON object of schema ctx.route/v1.

Rules:
- Decompose only where helpful. A trivial task is ONE node. Fan out only for independent subtasks or causal chains.
- Each node: {"id","goal","role","min_tier","needs":[tags],"deps":[ids],"est_input_tokens","est_output_tokens"}.
  Optional: "thinking_level" ("off"|"minimal"|"low"|"medium"|"high"), "host":"<name>", "model":"<id>", "prefer":"cheap"|"strong".
- min_tier (economy < standard < frontier) & thinking allocation:
    * exploration / search / triage / verify -> economy (thinking: "off")
    * SIMPLE edit / diff -> economy (Gemini-3.5-flash-lite, thinking: "off")
    * COMPLEX implementation / multi-file logic -> standard (Gemini-3.6-flash, thinking: "low")
    * PLANNING / architecture / hard reasoning -> frontier (prefer: "strong" -> Opus-4.8 / Sol, thinking: "adaptive"|"low")
- deps make a node wait for others; upstream evidence is handed over via a compact checkpoint: digest.
- Keep the graph acyclic and bounded (<= 12 nodes).
Return: {"schema":"ctx.route/v1","nodes":[ ... ]}
```

---

## 2. JSON Schema (`ctx.route/v1`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "UnifiedRoutePlanSchema",
  "type": "object",
  "required": ["schema", "nodes"],
  "properties": {
    "schema": {
      "type": "string",
      "enum": ["ctx.route/v1"]
    },
    "nodes": {
      "type": "array",
      "minItems": 1,
      "maxItems": 12,
      "items": {
        "$ref": "#/definitions/RouteNode"
      }
    }
  },
  "definitions": {
    "RouteNode": {
      "type": "object",
      "required": ["id", "goal", "role", "min_tier", "deps"],
      "properties": {
        "id": { "type": "string" },
        "goal": { "type": "string" },
        "role": {
          "type": "string",
          "enum": ["explore", "search", "plan", "architect", "reason", "implement", "edit", "code", "verify", "test", "triage", "review", "task"]
        },
        "min_tier": {
          "type": "string",
          "enum": ["economy", "standard", "frontier"]
        },
        "thinking_level": {
          "type": "string",
          "enum": ["off", "minimal", "low", "medium", "high"],
          "default": "off"
        },
        "needs": {
          "type": "array",
          "items": { "type": "string" }
        },
        "deps": {
          "type": "array",
          "items": { "type": "string" }
        },
        "est_input_tokens": { "type": "integer", "default": 20000 },
        "est_output_tokens": { "type": "integer", "default": 3000 },
        "host": { "type": "string" },
        "model": { "type": "string" },
        "prefer": {
          "type": "string",
          "enum": ["cheap", "strong"],
          "default": "cheap"
        }
      }
    }
  }
}
```
