---
id: refactoring
name: Refactoring
description: Safe structural improvement with characterization tests.
always: false
priority: 50
tags: [refactor, cleanup, duplicate]
triggers: [refactor, cleanup, duplicate, simplify]
---

# Refactoring

- Only refactor with a clear benefit and a way to verify.
- Prefer characterization tests before moving behavior.
- Delete dead code you introduced; leave pre-existing dead code unless asked.
- Keep behavior identical unless the goal says otherwise.
