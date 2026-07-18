---
id: software-engineering
name: Software Engineering Core
description: Production-grade SE workflow — scout, plan, implement, verify, deliver.
always: true
priority: 1
tags: [engineering, coding, se]
triggers: [implement, fix, build, refactor, feature, bug]
---

# Software Engineering Core

## Protocol
1. **Scout** — index/search/read until you understand the real codepaths.
2. **Criteria** — write explicit success checks (commands/tests/assertions).
3. **Plan** — 3–8 concrete steps; smallest viable change.
4. **Implement** — surgical edits; match existing style; no drive-by refactors.
5. **Verify** — run the success checks; read output; iterate.
6. **Deliver** — summarize paths changed + evidence of verification.

## Patch quality (benchmark-winning habits)
- Fail then fix: for bugs, reproduce first.
- Prefer `fs_edit` over wholesale rewrites.
- Keep unrelated files untouched.
- After a failing command, change approach based on the error — do not repeat blindly.
- When tests pass, stop. Do not "improve" further unless asked.
