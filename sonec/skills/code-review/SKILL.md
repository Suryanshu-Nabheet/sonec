---
id: code-review
name: Code Review
description: Aggressive production review — correctness, security, API design, tests.
always: false
priority: 30
tags: [review, pr, audit]
triggers: [review, critique, audit, pr]
---

# Code Review

Review like a staff engineer. Prefer findings with file:line evidence.

Checklist:
- Correctness & edge cases
- Security (injection, secrets, authz)
- API consistency & naming
- Error handling
- Tests adequate for the risk
- Dead code / duplication
- Performance only with evidence

Output a severity-ordered table: Critical / High / Medium / Low.
