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
# add info logger



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
from typing import Dict, Optional, List, Any
from verl import DataProto

# =========================================================================
# 1. First Turn Cleaning
# =========================================================================
def get_first_turn_batch(batch: DataProto, tokenizer) -> DataProto:
    """
    [Step 1: Data Cleaning]
    Batch In -> Batch Out
    
    Function: Logically truncates content after the first <|im_end|> token by 
    modifying the Attention Mask. It does NOT modify the physical shape or 
    data of input_ids or responses.
    
    Args:
        batch: Original DataProto
        tokenizer: Used to retrieve the im_end_id
    
    Returns:
        DataProto: A deep-copied Batch with modified masks.
    """
    # 1. Deep copy to prevent side effects on the original training data
    new_batch = copy.deepcopy(batch)
    device = new_batch.batch.device
    
    # 2. Retrieve <|im_end|> token id
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
         im_end_id = tokenizer.eos_token_id
    else:
         im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    if isinstance(im_end_id, list): im_end_id = im_end_id[0]

    # 3. Extract data
    responses = new_batch.batch["responses"]
    B, R_len = responses.shape
    
    # 4. Vectorized search for truncation points
    is_end = (responses == im_end_id)
    end_indices = is_end.float().argmax(dim=1) 
    has_end = is_end.sum(dim=1) > 0
    
    # Determine valid length: index + 1 if found, else full length
    valid_lens = torch.where(has_end, end_indices + 1, torch.tensor(R_len, device=device))
    
    # 5. Generate Tail Mask
    col_indices = torch.arange(R_len, device=device).unsqueeze(0) 
    cutoff = valid_lens.unsqueeze(1)
    
    # 1 where index < valid_len, else 0
    tail_mask = (col_indices < cutoff).long()
    
    # 6. Update Masks in Batch
    if "attention_mask" in new_batch.batch:
        new_batch.batch["attention_mask"][:, -R_len:] *= tail_mask
    if "response_mask" in new_batch.batch:
         new_batch.batch["response_mask"] *= tail_mask
    if "loss_mask" in new_batch.batch:
         new_batch.batch["loss_mask"] *= tail_mask

    return new_batch


# =========================================================================
# 2. Compute MI (Memory Optimized with Chunking)
# =========================================================================
def compute_mi(batch: DataProto, actor_wg) -> Dict[str, float]:
    """
    [Step 2: Memory-Efficient Cross-Scoring]
    
    Strategy:
    Instead of constructing a massive M*N batch (which causes OOM), 
    we iterate through N unique prompts. In each iteration, we construct 
    a batch of size M (pairing 1 Prompt with M Responses).
    
    Complexity:
    - Time: Same as full broadcast (total computations unchanged).
    - Memory: Reduced by factor of N (Batch size stays M).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- A. Basic Information Extraction ---
    original_responses = batch.batch["responses"]   # [M, R]
    original_input_ids = batch.batch["input_ids"]   # [M, P+R]
    original_pos_ids = batch.batch.get("position_ids", None)
    
    # Extract Multi-Modal Inputs (e.g., images/videos)
    multi_modal_inputs = batch.non_tensor_batch.get("multi_modal_inputs", None)
    has_multi_modal = multi_modal_inputs is not None
    
    M_responses, R_len = original_responses.shape
    P_len = original_input_ids.shape[1] - R_len 
    
    # Get the Cleaned Response Mask (from Step 1)
    clean_response_mask = batch.batch["attention_mask"][:, -R_len:] # [M, R]
    
    # Get Group Info
    group_ids = batch.non_tensor_batch.get("group_idx") or batch.non_tensor_batch.get("uid")
    if group_ids is None: group_ids = np.arange(M_responses)
    unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
    N_prompts = len(unique_groups)
    
    if N_prompts < 2: 
        return {}

    # Check for Complex Position IDs (e.g., Qwen2-VL has 3D/4D pos_ids)
    is_complex_pos = (original_pos_ids is not None) and (original_pos_ids.ndim > 2)

    # --- B. Prepare Result Matrix ---
    # matrix_log_probs[j, i] = LogProb(Response i | Prompt j)
    # We will fill this matrix row by row.
    matrix_log_probs = torch.zeros((N_prompts, M_responses), device=device, dtype=torch.float32)

    # --- C. Pre-fetch Prompt Metadata ---
    prompt_indices = [] # Indices of the first sample for each group
    for gid in unique_groups:
        idx = np.where(group_ids == gid)[0][0]
        prompt_indices.append(idx)
    
    # Convert to Tensor for slicing
    # unique_prompt_input_ids: [N, P]
    unique_prompt_input_ids = original_input_ids[prompt_indices, :P_len]
    
    unique_prompt_pos_ids = None
    if is_complex_pos:
        # unique_prompt_pos_ids: [N, ..., P]
        unique_prompt_pos_ids = original_pos_ids[prompt_indices, ..., :P_len]

    # --- D. Chunking Loop (Iterate by Target Prompt) ---
    for j in range(N_prompts):
        
        # 1. Prepare Current Prompt (Expand 1 -> M)
        # curr_prompt_ids: [1, P] -> [M, P]
        curr_prompt_ids = unique_prompt_input_ids[j:j+1].expand(M_responses, -1)
        
        # Concatenate Input IDs: [M, P+R]
        cross_input_ids = torch.cat([curr_prompt_ids, original_responses], dim=1)
        
        # 2. Prepare Masks
        prompt_mask = (curr_prompt_ids != 0).long()
        # Note: Response mask is reused from the M original responses
        cross_attention_mask = torch.cat([prompt_mask, clean_response_mask], dim=1)
        
        # 3. Handle Position IDs (The tricky part)
        if is_complex_pos:
            # === Anchor Shifting for Qwen2-VL ===
            
            # a. Expand Current Prompt Pos: [1, ..., P] -> [M, ..., P]
            # (Assuming dims are [Batch, Channels, Seq])
            curr_prompt_pos = unique_prompt_pos_ids[j:j+1].expand(M_responses, -1, -1)
            
            # b. Calculate Shift
            # New Anchor: The end position of the current target prompt
            new_anchor = curr_prompt_pos[..., -1:] # [M, ..., 1]
            
            # Old Anchors: The end position of the ORIGINAL prompts for these M responses
            # We slice the prompt part from original_pos_ids and take the last token
            old_anchors = original_pos_ids[..., :P_len][..., -1:] # [M, ..., 1]
            
            # Shift = New - Old
            shift = new_anchor - old_anchors # [M, ..., 1]
            
            # c. Apply Shift to Response Pos
            # Take original response positions and shift them
            original_response_pos = original_pos_ids[..., -R_len:]
            curr_response_pos = original_response_pos + shift
            
            # d. Concatenate
            cross_position_ids = torch.cat([curr_prompt_pos, curr_response_pos], dim=-1)
            
        else:
            # === Simple Regeneration (Standard LLMs) ===
            cross_position_ids = torch.cumsum(cross_attention_mask, dim=1) - 1
            cross_position_ids.masked_fill_(cross_attention_mask == 0, 0)
            
        # 4. Handle Multi-Modal Inputs
        # We need to broadcast the image/video from Prompt j to all M responses
        cross_non_tensor_batch = {}
        if has_multi_modal:
            # Get the single image object for the current prompt group
            src_idx = prompt_indices[j]
            single_image_data = multi_modal_inputs[src_idx]
            
            # Replicate M times (Reference Copy, low memory overhead)
            if isinstance(multi_modal_inputs, np.ndarray):
                # Create object array filled with the same image object
                cross_imgs = np.array([single_image_data] * M_responses, dtype=object)
            else:
                cross_imgs = [single_image_data] * M_responses
                
            cross_non_tensor_batch["multi_modal_inputs"] = cross_imgs

        # 5. Construct DataProto (Batch Size = M)
        cross_batch = DataProto.from_dict(
            tensors={
                "input_ids": cross_input_ids,
                "responses": original_responses,
                "attention_mask": cross_attention_mask,
                "position_ids": cross_position_ids
            },
            non_tensors=cross_non_tensor_batch,
            meta_info=batch.meta_info # Inherit config like temperature/micro_batch_size
        )
        
        # 6. Compute Log Probabilities
        # The actor handles micro-batching internally for these M samples
        log_probs_raw, _ = actor_wg.compute_log_prob(cross_batch)
        
        if log_probs_raw.shape[1] != R_len:
            log_probs_raw = log_probs_raw[:, -R_len:]
            
        # 7. Store Scores
        # Mask out truncated parts (garbage after im_end)
        valid_log_probs = log_probs_raw * clean_response_mask # [M, R]
        seq_log_probs = valid_log_probs.sum(dim=-1)           # [M]
        
        # Fill the j-th row of the matrix
        matrix_log_probs[j, :] = seq_log_probs

    # --- E. Compute MI Metrics ---
    # Transpose Matrix to [M, N] -> (Response i, Prompt j)
    matrix_log_probs = matrix_log_probs.t() 
    
    # Matched: Diagonal logic (Response i belongs to its original group)
    matched = matrix_log_probs[torch.arange(M_responses), group_indices]
    
    # Marginal: LogSumExp over all N Prompts (Normalized)
    marginal = torch.logsumexp(matrix_log_probs, dim=1) - math.log(N_prompts)
    
    mi_per_sample = matched - marginal
    
    # --- F. Aggregate Results ---
    group_mi_results = {}
    mi_numpy = mi_per_sample.detach().cpu().numpy()
    
    # Global Average
    group_mi_results["collapse/mi_global"] = float(np.mean(mi_numpy))
    
    # Per-Group Average
    for gid in unique_groups:
        mask = (group_ids == gid)
        if mask.any():
            group_mi_results[f"collapse/mi_group_{gid}"] = float(np.mean(mi_numpy[mask]))
            
    return group_mi_results


# =========================================================================
# 3. Entry Point
# =========================================================================
def compute_group_mi_first_turn(batch: DataProto, actor_wg, tokenizer) -> Dict[str, float]:
    """
    Main Entry Point.
    
    Pipeline:
    1. Clean Batch: Logically truncate to first turn (O(M)).
    2. Compute MI: Perform chunked cross-scoring (O(M*N)).
    """
    cleaned_batch = get_first_turn_batch(batch, tokenizer)
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
