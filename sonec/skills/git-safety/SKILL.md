---
id: git-safety
name: Git Safety
description: Safe git operations — no auto commit/push; conventional commits on request.
always: false
priority: 40
tags: [git, commit, push, branch]
triggers: [git, commit, push, branch, pr, pull request]
---

# Git Safety

- Never commit or push unless explicitly requested.
- Never `--force` push to main/master.
- Never rewrite published history without explicit approval.
- Never commit secrets (`.env`, keys, credentials).
- On request: status + diff + log style → stage → commit via HEREDOC → verify status.
- Prefer focused commits over mixed dumps.
