"""
Per-group metrics for plotting correlation analysis.

Each metric function takes a DataProto batch and returns Dict[group_id, float].
"""

import math
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


def _get_first_turn_mask(responses: torch.Tensor, response_mask: torch.Tensor, im_end_token_id: int) -> torch.Tensor:
    """Create a mask that only covers the first turn (up to first <|im_end|> token).

    Args:
        responses: Response token ids, shape (batch_size, seq_len)
        response_mask: Original response mask, shape (batch_size, seq_len)
        im_end_token_id: Token id for <|im_end|>

    Returns:
        Modified mask that zeros out everything after the first <|im_end|> token.
    """
    batch_size, seq_len = responses.shape
    first_turn_mask = response_mask.clone()

    for i in range(batch_size):
        # Find positions where token equals im_end_token_id
        im_end_positions = (responses[i] == im_end_token_id).nonzero(as_tuple=True)[0]

        if len(im_end_positions) > 0:
            # Get the first im_end position
            first_im_end_pos = im_end_positions[0].item()
            # Zero out everything after the first im_end (exclusive, keep the im_end token)
            if first_im_end_pos + 1 < seq_len:
                first_turn_mask[i, first_im_end_pos + 1:] = 0

    return first_turn_mask


def group_mi_estimate(batch: DataProto, tokenizer=None) -> Dict[str, float]:
    """Compute per-group Mutual Information (MI) estimate for the FIRST TURN only.

    MI estimates how much the responses depend on their specific prompts.
    High MI indicates prompt-specific responses (healthy).
    Low MI indicates template collapse (responses are similar regardless of prompt).

    Formula: MI = E[log p(r|x) - log p_mix(r)]
    where:
    - log p(r|x) = matched log prob (response under its true prompt)
    - log p_mix(r) = marginal log prob (mixture over all prompts)

    This function only considers the first turn of the response (up to the first
    <|im_end|> token) to focus on the initial reasoning step.

    Since we don't have cross-scoring (which would require running the model
    N times), we approximate the marginal using the group-level mean log probs
    as representatives of each prompt's likelihood.

    Args:
        batch: DataProto containing old_log_probs, response_mask, responses, and group identifiers.
        tokenizer: Optional tokenizer to get im_end token id. Defaults to Qwen2-VL tokenizer.

    Returns:
        Dict mapping group_id to MI estimate for that group.
    """
    group_ids = _get_group_ids(batch)

    if "old_log_probs" not in batch.batch:
        raise ValueError("Batch must contain 'old_log_probs' for MI computation")

    old_log_probs = batch.batch["old_log_probs"]
    response_mask = batch.batch["response_mask"]
    responses = batch.batch["responses"]

    # Get im_end token id
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)

    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Convert to torch tensors if needed
    if not isinstance(old_log_probs, torch.Tensor):
        old_log_probs = torch.tensor(old_log_probs)
    if not isinstance(response_mask, torch.Tensor):
        response_mask = torch.tensor(response_mask)
    if not isinstance(responses, torch.Tensor):
        responses = torch.tensor(responses)

    # Create first-turn-only mask
    first_turn_mask = _get_first_turn_mask(responses, response_mask, im_end_token_id)

    # Move to CPU for computation
    old_log_probs = old_log_probs.detach().cpu()
    first_turn_mask = first_turn_mask.detach().cpu()

    # Compute per-sample mean log prob (normalized by first turn length)
    mask_sum = first_turn_mask.sum(dim=-1).clamp(min=1e-8)
    sample_log_probs = (old_log_probs * first_turn_mask).sum(dim=-1) / mask_sum
    sample_log_probs = sample_log_probs.numpy()

    # Get unique groups
    unique_groups = np.unique(group_ids)
    N = len(unique_groups)

    if N < 2:
        # Need at least 2 groups for meaningful MI
        return {str(gid): 0.0 for gid in np.unique(group_ids)}

    # Step 1: Compute mean log prob for each group (as group representative)
    group_to_samples = defaultdict(list)
    for i, gid in enumerate(group_ids):
        group_to_samples[str(gid)].append(sample_log_probs[i])

    group_mean_log_probs = {}
    for gid, lps in group_to_samples.items():
        group_mean_log_probs[gid] = np.mean(lps)

    # Step 2: Compute marginal for each sample using logsumexp over group means
    # marginal(r) ≈ logsumexp(group_mean_log_probs) - log(N)
    # This approximates the mixture: p_mix(r) = (1/N) * sum_j p(r|x_j)
    group_means_array = np.array(list(group_mean_log_probs.values()))

    # For each sample, compute its MI contribution
    # matched[i] = sample_log_probs[i]
    # marginal[i] ≈ logsumexp(group_means) - log(N)
    # But we want per-sample marginal, so we use logsumexp properly

    # The marginal for sample i is: log(1/N * sum_j exp(log p(r_i|x_j)))
    # Since we don't have cross-scores, we approximate:
    # For sample i in group g, we use group_mean_log_probs as proxy for cross-scores

    # Per-sample marginal using group means as proxies
    marginal_global = np.logaddexp.reduce(group_means_array) - math.log(N)

    # Step 3: Compute per-group MI
    # For each sample: mi = matched - marginal
    # For each group: mean over samples in that group
    group_mi = defaultdict(list)
    for i, gid in enumerate(group_ids):
        matched = sample_log_probs[i]
        # MI contribution for this sample
        mi_sample = matched - marginal_global
        group_mi[str(gid)].append(mi_sample)

    result = {}
    for gid, mis in group_mi.items():
        result[gid] = float(np.mean(mis))

    return result


# Registry of all available per-group metrics
REGISTERED_METRICS = {
    "reward/variance": group_reward_variance,
    # "reward/mean": group_reward_mean,
    "actor/entropy": group_entropy_mean,
    "response/length": group_response_len_mean,
    # "advantage/mean": group_advantage_mean,
    # "actor/log_prob": group_old_log_prob_mean,
    "collapse/mi": group_mi_estimate,
}
