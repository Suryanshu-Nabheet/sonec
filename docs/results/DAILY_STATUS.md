# Daily status

Generated: `2026-07-19T10:51:14Z`

**Overall:** PASS

| Check | OK | Detail |
| --- | --- | --- |
| `capabilitybench_shape` | yes | tasks=200 by_diff={'easy': 70, 'medium': 70, 'hard': 60} cats=10 |
| `pytest` | yes | 84 passed in 0.50s |
| `mock_bench` | yes | suite=ab_agent_2b_hard.json passed=8/8 |

## CapabilityBench

- Tasks: **200** (sealed)
- Difficulty: easy=70 medium=70 hard=60

## Published smoke snapshot (from repo files)

- Compare winner: `None`
- Board winner: `sonec`

Live Cap200 / MLX A/B is run locally (`SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh`).
This daily job validates the codebase and sealed suite on GitHub Actions.
