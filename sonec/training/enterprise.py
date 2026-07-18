"""Deprecated alias — use sonec.training.specialize."""

from sonec.training.specialize import (  # noqa: F401
    TrainReport,
    assemble_sft_corpus,
    generate_gold_agent_examples,
    prepare_rollout_fuel,
    run_enterprise_train,
    run_train_step,
)

__all__ = [
    "TrainReport",
    "assemble_sft_corpus",
    "generate_gold_agent_examples",
    "prepare_rollout_fuel",
    "run_enterprise_train",
    "run_train_step",
]
