# Harness freeze (Phases 0–2)

Single production runtime for CLI, eval, rollouts, serve, and MCP.

- Evidence-only success (graders, not model self-report)
- Thin always-on prompt; core tools hashed
- Trajectory JSONL with `harness_version` + `tool_schema_hash`

Used to specialize **sonec** from `qwen3.5:2b`. Product status: [`GATE_REPORT_MODEL_STACK.md`](GATE_REPORT_MODEL_STACK.md).
