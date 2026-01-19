"""
Per-group metrics for plotting correlation analysis.

Each metric function takes a DataProto batch and returns Dict[group_id, float].
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict
from verl import DataProto


def _get_group_ids(batch: DataProto) -> np.ndarray:
    """Get group identifiers from batch."""
    if "group_idx" in batch.non_tensor_batch:
        return batch.non_tensor_batch["group_idx"]
    elif "uid" in batch.non_tensor_batch:
        return batch.non_tensor_batch["uid"]
    else:
        raise ValueError("Batch must contain 'group_idx' or 'uid' in non_tensor_batch")


def group_reward_variance(batch: DataProto, ddof: int = 0) -> Dict[str, float]:
    """Compute the variance of rewards within each group.

    For each group, calculates Var(total_reward_i) for all samples i in that group.

    Args:
        batch: DataProto containing token_level_scores and group identifiers.
        ddof: Delta degrees of freedom for variance calculation.

    Returns:
        Dict mapping group_id to variance of total rewards within that group.
    """
    token_level_scores = batch.batch["token_level_scores"]
    group_ids = _get_group_ids(batch)

    # Compute total reward per sample
    if isinstance(token_level_scores, torch.Tensor):
        total_rewards = token_level_scores.sum(dim=-1).detach().cpu().numpy()
    else:
        total_rewards = np.asarray(token_level_scores).sum(axis=-1)

    # Group rewards by group_id
    group_rewards = defaultdict(list)
    for gid, reward in zip(group_ids, total_rewards):
        group_rewards[str(gid)].append(float(reward))

    # Compute variance for each group
    result = {}
    for gid, rewards in group_rewards.items():
        if len(rewards) <= 1:
            result[gid] = 0.0
        else:
            result[gid] = float(np.var(rewards, ddof=ddof))

    return result


def group_reward_mean(batch: DataProto) -> Dict[str, float]:
    """Compute the mean of rewards within each group.

    Args:
        batch: DataProto containing token_level_scores and group identifiers.

    Returns:
        Dict mapping group_id to mean of total rewards within that group.
    """
    token_level_scores = batch.batch["token_level_scores"]
    group_ids = _get_group_ids(batch)

    # Compute total reward per sample
    if isinstance(token_level_scores, torch.Tensor):
        total_rewards = token_level_scores.sum(dim=-1).detach().cpu().numpy()
    else:
        total_rewards = np.asarray(token_level_scores).sum(axis=-1)

    # Group rewards by group_id
    group_rewards = defaultdict(list)
    for gid, reward in zip(group_ids, total_rewards):
        group_rewards[str(gid)].append(float(reward))

    # Compute mean for each group
    result = {}
    for gid, rewards in group_rewards.items():
        result[gid] = float(np.mean(rewards))

    return result


def group_entropy_mean(batch: DataProto) -> Dict[str, float]:
    """Compute the mean entropy across samples within each group.

    Uses pre-computed per-sample entropys aggregated by response mask.
    Entropy values should be in batch.batch["entropys"] (per-token) or
    batch.batch["sample_entropys"] (pre-aggregated per sample).

    Args:
        batch: DataProto containing entropys and response_mask.

    Returns:
        Dict mapping group_id to mean entropy within that group.
    """
    group_ids = _get_group_ids(batch)

    # Check for pre-aggregated sample entropys first
    if "sample_entropys" in batch.batch:
        sample_entropys = batch.batch["sample_entropys"]
        if isinstance(sample_entropys, torch.Tensor):
            sample_entropys = sample_entropys.detach().cpu().numpy()
    elif "entropys" in batch.batch:
        # Compute per-sample entropy from token-level entropys
        entropys = batch.batch["entropys"]
        response_mask = batch.batch["response_mask"]

        if isinstance(entropys, torch.Tensor):
            entropys = entropys.detach().cpu()
            response_mask = response_mask.detach().cpu()

            # Compute masked mean entropy per sample (token-mean aggregation)
            mask_sum = response_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sample_entropys = (entropys * response_mask).sum(dim=-1) / mask_sum.squeeze(-1)
            sample_entropys = sample_entropys.numpy()
        else:
            entropys = np.asarray(entropys)
            response_mask = np.asarray(response_mask)
            mask_sum = response_mask.sum(axis=-1, keepdims=True)
            mask_sum = np.maximum(mask_sum, 1e-8)
            sample_entropys = (entropys * response_mask).sum(axis=-1) / mask_sum.squeeze(-1)
    else:
        raise ValueError("Batch must contain 'entropys' or 'sample_entropys' for entropy computation")

    # Group entropys by group_id
    group_entropys = defaultdict(list)
    for gid, entropy in zip(group_ids, sample_entropys):
        group_entropys[str(gid)].append(float(entropy))

    # Compute mean entropy for each group
    result = {}
    for gid, entropys in group_entropys.items():
        result[gid] = float(np.mean(entropys))

    return result


def group_response_len_mean(batch: DataProto) -> Dict[str, float]:
    """Compute the mean response length across samples within each group.

    Response length is computed from the response_mask (sum of valid tokens).

    Args:
        batch: DataProto containing response_mask and group identifiers.

    Returns:
        Dict mapping group_id to mean response length within that group.
    """
    group_ids = _get_group_ids(batch)
    response_mask = batch.batch["response_mask"]

    # Compute response length per sample
    if isinstance(response_mask, torch.Tensor):
        response_lengths = response_mask.sum(dim=-1).detach().cpu().numpy()
    else:
        response_lengths = np.asarray(response_mask).sum(axis=-1)

    # Group by group_id
    group_lengths = defaultdict(list)
    for gid, length in zip(group_ids, response_lengths):
        group_lengths[str(gid)].append(float(length))

    # Compute mean length for each group
    result = {}
    for gid, lengths in group_lengths.items():
        result[gid] = float(np.mean(lengths))

    return result


def group_advantage_mean(batch: DataProto) -> Dict[str, float]:
    """Compute the mean advantage across samples within each group.

    Args:
        batch: DataProto containing advantages and response_mask.

    Returns:
        Dict mapping group_id to mean advantage within that group.
    """
    group_ids = _get_group_ids(batch)

    if "advantages" not in batch.batch:
        raise ValueError("Batch must contain 'advantages' for advantage computation")

    advantages = batch.batch["advantages"]
    response_mask = batch.batch["response_mask"]

    if isinstance(advantages, torch.Tensor):
        advantages = advantages.detach().cpu()
        response_mask = response_mask.detach().cpu()

        # Compute masked mean advantage per sample
        mask_sum = response_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sample_advantages = (advantages * response_mask).sum(dim=-1) / mask_sum.squeeze(-1)
        sample_advantages = sample_advantages.numpy()
    else:
        advantages = np.asarray(advantages)
        response_mask = np.asarray(response_mask)
        mask_sum = response_mask.sum(axis=-1, keepdims=True)
        mask_sum = np.maximum(mask_sum, 1e-8)
        sample_advantages = (advantages * response_mask).sum(axis=-1) / mask_sum.squeeze(-1)

    # Group advantages by group_id
    group_advantages = defaultdict(list)
    for gid, adv in zip(group_ids, sample_advantages):
        group_advantages[str(gid)].append(float(adv))

    # Compute mean advantage for each group
    result = {}
    for gid, advs in group_advantages.items():
        result[gid] = float(np.mean(advs))

    return result


def group_old_log_prob_mean(batch: DataProto) -> Dict[str, float]:
    """Compute the mean old log probability across samples within each group.

    Args:
        batch: DataProto containing old_log_probs and response_mask.

    Returns:
        Dict mapping group_id to mean old log prob within that group.
    """
    group_ids = _get_group_ids(batch)

    if "old_log_probs" not in batch.batch:
        raise ValueError("Batch must contain 'old_log_probs'")

    old_log_probs = batch.batch["old_log_probs"]
    response_mask = batch.batch["response_mask"]

    if isinstance(old_log_probs, torch.Tensor):
        old_log_probs = old_log_probs.detach().cpu()
        response_mask = response_mask.detach().cpu()

        # Compute masked mean log prob per sample
        mask_sum = response_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sample_log_probs = (old_log_probs * response_mask).sum(dim=-1) / mask_sum.squeeze(-1)
        sample_log_probs = sample_log_probs.numpy()
    else:
        old_log_probs = np.asarray(old_log_probs)
        response_mask = np.asarray(response_mask)
        mask_sum = response_mask.sum(axis=-1, keepdims=True)
        mask_sum = np.maximum(mask_sum, 1e-8)
        sample_log_probs = (old_log_probs * response_mask).sum(axis=-1) / mask_sum.squeeze(-1)

    # Group by group_id
    group_log_probs = defaultdict(list)
    for gid, lp in zip(group_ids, sample_log_probs):
        group_log_probs[str(gid)].append(float(lp))

    # Compute mean for each group
    result = {}
    for gid, lps in group_log_probs.items():
        result[gid] = float(np.mean(lps))

    return result


# Registry of all available per-group metrics
REGISTERED_METRICS = {
    "reward/variance": group_reward_variance,
    # "reward/mean": group_reward_mean,
    "actor/entropy": group_entropy_mean,
    "response/length": group_response_len_mean,
    # "advantage/mean": group_advantage_mean,
    # "actor/log_prob": group_old_log_prob_mean,
}
