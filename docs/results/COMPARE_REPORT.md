# sonec vs base — live compare

Generated: `2026-07-19T07:36:47Z`  
Suite: `examples/benchmarks/ab_agent_2b_hard.json`

| Arm | Kind | Pass rate | Passed | Mean score | Mean duration |
| --- | --- | --- | --- | --- | --- |
| sonec_lora | lora | 100% | 8/8 | 1.00 | 8.6s |
| qwen35_2b_base | base | 100% | 8/8 | 1.00 | 16.5s |

**Winner (pass_rate):** tie  
**Delta pass_rate (lora − base):** +0%  
**Speed:** sonec LoRA ≈ **1.9× faster** mean duration (8.6s vs 16.5s) on the same MLX base.

Same frozen harness; arms differ only by weights/endpoint (LoRA `:8080` vs base `:8081`).

Product sonec = LoRA adapter served via `sonec serve-llm`.

Per-task: all 8 hard tasks passed on both arms (`hard-nested-readme`, `hard-py-cli`, `hard-pkg-api`, `hard-fix-clamp`, `hard-json-flag`, `hard-verify-pair`, `hard-two-modules`, `hard-restraint`).
