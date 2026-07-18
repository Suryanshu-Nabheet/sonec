---
id: benchmark-swe
name: Benchmark SWE
description: SWE-bench / agentic coding contest posture — reproduce, patch, verify.
always: false
priority: 10
tags: [benchmark, swe-bench, eval, harness]
triggers: [benchmark, swe-bench, swebench, eval, leaderboard, score]
---

# Benchmark SWE Posture

Winning agentic coding evals is a process, not a prompt:

1. **Read the instance** — fail to understand the bug → fail the patch.
2. **Localize** — search symbols/tests related to the failure; open the right files.
3. **Reproduce** — run the failing test; save the signal.
4. **Minimal patch** — touch only what the test requires.
5. **Regression** — re-run failing tests + nearby suite.
6. **No gold-plating** — extra refactors lose score and burn budget.
7. **Budget** — track iterations; if stuck, re-localize rather than thrash.

Use the SONEC harness phases (recon → plan → execute → verify → critique).
The critic pass exists to catch "wrote files but tests still fail".
