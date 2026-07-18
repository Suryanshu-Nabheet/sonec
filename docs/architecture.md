# SONEC Architecture (v0.2)

## Purpose

SONEC turns a strong base model (default: Kimi K3) into a **benchmark-oriented coding agent** by surrounding it with a real harness: rules, skills, phased control, verification, and critique.

The model is a dependency. The harness is the product.

## Layout

```
sonec/                     # installable package (repo-root, not src/)
  harness/                 # orchestrator, context, critic
  rules/
    prebuilt/              # shipped operating rules (*.mdc)
    engine.py
  skills/*/SKILL.md        # progressive skills
  agent/                   # single-loop runtime (primitive)
  tools/ llm/ memory/ …    # adapters
```

## Control plane

```
goal
  → RulesEngine.active(goal) + SkillsRegistry.activate(goal)
  → ContextAssembler.build_system_prompt
  → AgenticOrchestrator phases:
        RECON → PLAN → EXECUTE → VERIFY → CRITIQUE → DELIVER
  → Critic gate (fail → recovery tool loop)
  → AgentRunResult
```

## Trade-offs

| Choice | Why | Cost |
| --- | --- | --- |
| Flat `sonec/` package | Clear mental model; no empty `src/` | Slightly less conventional than src-layout |
| Multi-phase harness default | Forces recon/verify discipline that wins evals | More LLM calls per task |
| Import prebuilt rules | Carry proven operator standards into the agent | Large prompt; truncated with `rules_load` |
| Skills progressive disclosure | Keep prompts sharp; deepen on demand | Router is heuristic (keyword/tag scoring) |
| Critic pass | Prevents false completion | Extra latency |

## Limitations

- Skill routing is lexical, not embedding-based (yet).
- Critic JSON must parse; falls back to fail-closed.
- Large prebuilt rules are truncated in the always-on prompt; full text via `rules_load`.

## Future

- Embedding skill retrieval
- Parallel specialist subagents (reviewer/tester)
- SWE-bench verified adapters in `sonec.eval`
- Streaming + token budgets per phase
