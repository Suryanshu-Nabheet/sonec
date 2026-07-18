# sonec vs base — live compare

Suite: `examples/benchmarks/ab_agent_v1.json`

| Arm | Kind | Pass rate | Passed | Mean score | Mean duration |
| --- | --- | --- | --- | --- | --- |
| sonec_lora | lora | 17% | 1/6 | 0.17 | 4.8s |
| qwen35_2b_base | base | 17% | 1/6 | 0.17 | 10.0s |

**Winner:** tie
**Delta pass_rate (lora − base):** +0%

Same frozen harness; arms differ only by weights/endpoint. Tie on pass_rate — inspect mean_score / per-task diffs.

Product sonec = LoRA adapter weights (`sonec serve-llm`), not a Modelfile SYSTEM string.
