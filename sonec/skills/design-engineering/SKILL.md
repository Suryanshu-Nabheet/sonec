---
id: design-engineering
name: Design Engineering
description: High-craft UI/motion — taste, purposeful animation, GPU-only motion.
always: false
priority: 35
tags: [design, ui, frontend, animation, motion]
triggers: [ui, frontend, animation, design, css, react, component]
---

# Design Engineering

- Motion must answer why (feedback, spatial consistency, state, anti-jank).
- Frequency gate: never animate 100+/day keyboard actions.
- Ease-out for enter/exit; never ease-in on UI; custom curves beat defaults.
- UI motion under 300ms; animate transform/opacity only.
- Never scale(0); origin-aware popovers; press feedback scale(0.97).
- prefers-reduced-motion: gentler, not zero.
- When reviewing UI, use a Before/After/Why markdown table.
