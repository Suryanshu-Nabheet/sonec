---
id: debugging
name: Debugging
description: Systematic debugging from traces, failing tests, and runtime evidence.
always: false
priority: 20
tags: [debug, bug, traceback, error]
triggers: [debug, bug, traceback, exception, failing, crash, error]
---

# Debugging

1. Capture the failing signal (test output, traceback, logs).
2. Parse the top frame and the origin frame.
3. Form one hypothesis; change one thing; re-run.
4. Prefer a minimal reproduction before a broad rewrite.
5. When fixed: keep the regression test that proved it.
