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


def group_reward_variance(batch: DataProto, ddof: int = 0, **kwargs) -> Dict[str, float]:
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


def group_entropy_mean(batch: DataProto, **kwargs) -> Dict[str, float]:
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


def group_response_len_mean(batch: DataProto,**kwargs) -> Dict[str, float]:
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








import copy
import math
import numpy as np
import torch
from typing import Dict, Optional
from verl import DataProto

def get_first_turn_batch(batch: DataProto, tokenizer) -> DataProto:
    """
    [Step 1: Data Cleaning]
    Batch In -> Batch Out
    
    Function: Logically truncates content after the first <|im_end|> token by only modifying 
    the Attention Mask. It does not modify the physical shape or data of input_ids or responses.
    
    Args:
        batch: Original DataProto
        tokenizer: Used to retrieve the im_end_id
    
    Returns:
        DataProto: A deep-copied Batch with modified masks.
    """
    # 1. Deep Copy to prevent side effects on the original training data
    new_batch = copy.deepcopy(batch)
    device = new_batch.batch.device
    
    # 2. Automatically retrieve the <|im_end|> token id
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
         im_end_id = tokenizer.eos_token_id
    else:
         # Fallback: try to retrieve by string conversion
         im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    if isinstance(im_end_id, list): im_end_id = im_end_id[0]

    # 3. Extract data
    responses = new_batch.batch["responses"] # [B, R_len]
    B, R_len = responses.shape
    
    # 4. Vectorized search for truncation points (First-Turn Finding)
    # is_end: [B, R_len] (bool)
    is_end = (responses == im_end_id)
    # end_indices: Index of the first True value for each sample (returns 0 if all False)
    end_indices = is_end.float().argmax(dim=1) 
    # has_end: Mark which samples actually contain im_end
    has_end = is_end.sum(dim=1) > 0
    
    # valid_lens: Valid length including im_end
    # If im_end is not found, default to full length (or non-zero length)
    valid_lens = torch.where(has_end, end_indices + 1, torch.tensor(R_len, device=device))
    
    # 5. Generate Tail Mask
    # Create column indices [1, R]
    col_indices = torch.arange(R_len, device=device).unsqueeze(0) 
    # Create truncation thresholds [B, 1]
    cutoff = valid_lens.unsqueeze(1)
    
    # Generate Mask: 1 where index < valid_len, else 0
    tail_mask = (col_indices < cutoff).long()
    
    # 6. Modify the Mask in the Batch
    # Only modify the last R_len columns (the Response part)
    if "attention_mask" in new_batch.batch:
        # Use *= to ensure originally padded positions (0) remain 0
        new_batch.batch["attention_mask"][:, -R_len:] *= tail_mask

    # Handle other potential masks
    if "response_mask" in new_batch.batch:
         new_batch.batch["response_mask"] *= tail_mask
    if "loss_mask" in new_batch.batch:
         new_batch.batch["loss_mask"] *= tail_mask

    # 7. Regenerate Position IDs (since tokens were masked out)
    # cumsum ensures position encodings remain compact
    if "attention_mask" in new_batch.batch:
        new_position_ids = torch.cumsum(new_batch.batch["attention_mask"], dim=1) - 1
        new_position_ids.masked_fill_(new_batch.batch["attention_mask"] == 0, 0)
        new_batch.batch["position_ids"] = new_position_ids

    return new_batch


def compute_mi(batch: DataProto, actor_wg) -> Dict[str, float]:
    """
    [Step 2: Cross-Scoring Calculation]
    
    Function: Computes Mutual Information on the cleaned mixed Batch and aggregates results by Group.
    Assumes the input batch is already in a First-Turn Cleaned state.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==========================
    # 1. Basic Information Extraction
    # ==========================
    original_responses = batch.batch["responses"]
    original_input_ids = batch.batch["input_ids"]
    M_responses, R_len = original_responses.shape
    P_len = original_input_ids.shape[1] - R_len # Length of Prompt
    
    # Get the cleaned Response Mask (reuse directly from batch)
    # This is crucial: we reuse the result from step 1 (get_first_turn_batch)
    clean_response_mask = batch.batch["attention_mask"][:, -R_len:]
    
    # Get Group Info
    group_ids = batch.non_tensor_batch.get("group_idx") or batch.non_tensor_batch.get("uid")
    if group_ids is None: group_ids = np.arange(M_responses)
    
    unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
    N_prompts = len(unique_groups)
    
    # Must have at least 2 different prompts to compute MI
    if N_prompts < 2: 
        return {}

    # ==========================
    # 2. Extract Unique Prompts
    # ==========================
    unique_prompts_list = []
    for gid in unique_groups:
        idx = np.where(group_ids == gid)[0][0]
        # Extract the left part (Prompt)
        unique_prompts_list.append(original_input_ids[idx, :P_len])
    unique_prompts = torch.stack(unique_prompts_list).to(device) # [N, P_len]
    
    # ==========================
    # 3. Construct Cross Batch (Broadcasting)
    # ==========================
    # Goal: Construct an M*N Batch
    
    # Expand Prompts: [N, P] -> [N*M, P] (Repeat Interleave: P1, P1, P2, P2...)
    expanded_prompts = unique_prompts.repeat_interleave(M_responses, dim=0)
    
    # Expand Responses: [M, R] -> [N*M, R] (Repeat: R1, R2, R3...)
    expanded_responses = original_responses.repeat(N_prompts, 1)
    
    # Expand Response Masks: [M, R] -> [N*M, R] (Synchronize with Response expansion)
    expanded_res_masks = clean_response_mask.repeat(N_prompts, 1)
    
    # Concatenate Input IDs
    cross_input_ids = torch.cat([expanded_prompts, expanded_responses], dim=1)
    
    # Generate Global Mask
    prompt_mask = (expanded_prompts != 0).long()
    cross_attention_mask = torch.cat([prompt_mask, expanded_res_masks], dim=1)
    
    # Generate Position IDs
    cross_position_ids = torch.cumsum(cross_attention_mask, dim=1) - 1
    cross_position_ids.masked_fill_(cross_attention_mask == 0, 0)
    
    # ==========================
    # 4. Actor Forward Pass
    # ==========================
    cross_batch = DataProto.from_dict(
        tensors={
            "input_ids": cross_input_ids,
            "responses": expanded_responses,
            "attention_mask": cross_attention_mask,
            "position_ids": cross_position_ids
        },
        meta_info=batch.meta_info
    )
    
    # Compute LogProb
    # Note: actor_wg.compute_log_prob usually handles micro-batching automatically
    log_probs_raw, _ = actor_wg.compute_log_prob(cross_batch)
    
    if log_probs_raw.shape[1] != R_len:
        log_probs_raw = log_probs_raw[:, -R_len:]
        
    # ==========================
    # 5. MI Formula Calculation
    # ==========================
    # Use the mask again to filter out LogProb noise from truncated parts (garbage after im_end)
    valid_log_probs = log_probs_raw * expanded_res_masks
    
    # Sum: [N*M]
    seq_log_probs = valid_log_probs.sum(dim=-1)
    
    # Reshape: [N, M] -> [M, N]
    # Matrix[i, j] = LogProb(Response i | Prompt j)
    # Note transpose: because we constructed with P_outer, R_inner
    matrix_log_probs = seq_log_probs.view(N_prompts, M_responses).t()
    
    # Matched: Diagonal logic (Each Response i corresponds to its real Group)
    matched = matrix_log_probs[torch.arange(M_responses), group_indices]
    
    # Marginal: LogSumExp over Prompts (Normalized to probability space)
    # log( 1/N * sum_j exp(L_ij) ) = logsumexp - log(N)
    marginal = torch.logsumexp(matrix_log_probs, dim=1) - math.log(N_prompts)
    
    mi_per_sample = matched - marginal
    
    # ==========================
    # 6. Aggregate Results by Group
    # ==========================
    group_mi_results = {}
    mi_numpy = mi_per_sample.detach().cpu().numpy()
    
    # Record global average
    group_mi_results["collapse/mi_global"] = float(np.mean(mi_numpy))
    
    # Record average per Group
    for gid in unique_groups:
        mask = (group_ids == gid)
        if mask.any():
            # Use str(gid) as key
            group_mi_results[f"collapse/mi_group_{gid}"] = float(np.mean(mi_numpy[mask]))
            
    return group_mi_results


def compute_group_mi_first_turn(batch: DataProto, actor_wg, tokenizer, **kwargs) -> Dict[str, float]:
    """
    [Entry Point] 
    
    Args:
        batch: Original DataProto (containing full multi-turn conversations)
        actor_wg: Actor WorkerGroup (used for calculating log_prob)
        tokenizer: Used to identify the im_end token
        
    Returns:
        Dict: {group_id: mi_value}
    """
    # 1. Preprocessing: Clean Batch, truncate to First Turn (O(M))
    # This function does not change physical data shape, only modifies the mask
    cleaned_batch = get_first_turn_batch(batch, tokenizer)
    
    # 2. Calculation: Cross-Scoring on the mixed Batch (O(N*M))
    # And split results by Group at the end
    mi_metrics = compute_mi(cleaned_batch, actor_wg)
    
    return mi_metrics


# Registry of all available per-group metrics
REGISTERED_METRICS = {
    "reward/variance": group_reward_variance,
    # "reward/mean": group_reward_mean,
    "actor/entropy": group_entropy_mean,
    "response/length": group_response_len_mean,
    # "advantage/mean": group_advantage_mean,
    # "actor/log_prob": group_old_log_prob_mean,
    "collapse/mi": compute_group_mi_first_turn,
}
