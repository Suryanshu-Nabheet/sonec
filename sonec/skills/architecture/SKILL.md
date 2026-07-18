---
id: architecture
name: Architecture Analysis
description: Dependency graphs, cycles, cohesion — change with intent.
always: false
priority: 45
tags: [architecture, design, module, dependency]
triggers: [architecture, module, dependency, redesign, layer]
---

# Architecture

Before large changes:
1. Map modules and edges.
2. Find cycles and high fan-out hubs.
3. Prefer extending existing abstractions over parallel ones.
4. Explain trade-offs; then implement the chosen option cleanly.
5. Keep folders purposeful; avoid deep nesting.
