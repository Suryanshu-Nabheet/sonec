# 2B-class agent leaderboard

Suite: `examples/benchmarks/ab_agent_v1.json`

**Winner:** `sonec`

| Rank | Model | Kind | Pass rate | Passed | Mean score | Mean duration |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | sonec | lora | 100% | 6/6 | 1.00 | 10.6s |
| 2 | qwen3.5-2b | base | 100% | 6/6 | 1.00 | 8.4s |
| 3 | llama3.2-3b | external | 67% | 4/6 | 0.75 | 3.0s |
| 4 | llama3.2-1b | external | 17% | 1/6 | 0.22 | 1.2s |
| 5 | qwen2.5-coder-1.5b | external | 17% | 1/6 | 0.22 | 1.2s |
| 6 | qwen2.5-1.5b | external | 17% | 1/6 | 0.22 | 2.2s |
| 7 | gemma2-2b | external | 0% | 0/6 | 0.00 | 0.0s |
| 8 | phi3-mini | external | 0% | 0/6 | 0.00 | 0.0s |
| 9 | deepseek-coder-1.3b | external | 0% | 0/6 | 0.00 | 0.0s |

Ranked by pass_rate, then mean_score, then speed. Same frozen harness.

