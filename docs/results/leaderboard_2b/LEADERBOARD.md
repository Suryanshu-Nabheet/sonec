# 2B-class agent leaderboard

Generated: `2026-07-19T07:39:54Z`  
Suite: `examples/benchmarks/ab_agent_2b_hard.json`

**Winner:** `sonec`

| Rank | Model | Kind | Pass rate | Passed | Mean score | Mean duration |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | sonec | lora | 100% | 8/8 | 1.00 | 8.5s |
| 2 | qwen3.5-2b | base | 100% | 8/8 | 1.00 | 11.5s |
| 3 | gemma2-2b | external | 0% | 0/8 | 0.00 | 0.0s |
| 4 | codegemma-2b | external | 0% | 0/8 | 0.00 | 0.0s |

Ranked by pass_rate, then mean_score, then speed. Same frozen harness.

Peers are strict ~2B only (`configs/leaderboard/arms_2b.json`). Smoke suite is saturated for tool-capable 2B arms — next decision metric is CapabilityBench (200).
