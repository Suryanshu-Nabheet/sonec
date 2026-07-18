---
id: testing-tdd
name: Testing & TDD
description: Goal-driven verification — failing tests first, then make them pass.
always: false
priority: 25
tags: [test, pytest, tdd, coverage]
triggers: [test, pytest, unittest, coverage, tdd, verify]
---

# Testing & TDD

Transform tasks into verifiable goals:
- "Add validation" → tests for invalid inputs, then make them pass
- "Fix the bug" → failing reproduction test, then make it pass

Prefer:
- Unit tests for pure logic
- Integration for tool/IO boundaries
- One assertion idea per test when clarity suffers

Never claim green without running the suite (or explaining why it could not run).
