# sonec vs base — live compare

Suite: `examples/benchmarks/ab_agent_v1.json`

| Arm | Kind | Pass rate | Passed | Mean score | Mean duration |
| --- | --- | --- | --- | --- | --- |
| sonec_lora | lora | 67% | 4/6 | 0.67 | 19.8s |
| qwen35_2b_base | base | 50% | 3/6 | 0.50 | 21.6s |

**Winner:** sonec_lora
**Delta pass_rate (lora − base):** +17%

Same frozen harness; arms differ only by weights/endpoint.

Product sonec = LoRA adapter served via sonec serve-llm.
