"""Bi-level GAE implementation."""

from vagen.algorithms._common import AdvantageInputs, advantage_estimator, register_algorithm
from vagen.algorithms._common.trajectory_algos import _compute_bi_level_gae


@advantage_estimator("bi_level_gae", needs_critic=True, turn_lumped_reward=True)
def compute_bi_level_gae(inputs: AdvantageInputs):
    return _compute_bi_level_gae(inputs)

SPEC = register_algorithm("bi_level_gae", compute_bi_level_gae)

__all__ = ["SPEC", "compute_bi_level_gae"]
