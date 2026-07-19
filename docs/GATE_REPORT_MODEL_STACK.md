# Gate — sonec specialization

**Product:** LoRA adapter `sonec` by Suryanshu Nabheet — coding model  
**Weights:** `artifacts/train/checkpoints/sonec-sft-mlx`  
**License:** Apache-2.0 — LICENSE · lineage — NOTICE

## Pipeline

1. TrainBench graded rollouts (training-only ids)
2. Verified live trajectories (path-correct writes, verify, restraint)
3. MLX LoRA SFT
4. Rejection sampling / group winners
5. Serve with `sonec serve-llm`

## Commands

```bash
sonec train --step --live-fuel --sft-iters 300 --gold-n 0 --train-n 40
sonec weights
sonec serve-llm --port 8080
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
sonec compare --out docs/results
```

Promote an adapter only when `sonec compare` shows a higher pass rate without restraint regression.
