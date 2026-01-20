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
import logging

logger = logging.getLogger(__name__)



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
import logging
from typing import Dict, Optional, List, Any
from verl import DataProto

# Initialize logger for error reporting
logger = logging.getLogger(__name__)

# =========================================================================
# 1. First Turn Cleaning (Physical Padding + Masking)
# =========================================================================
def get_first_turn_batch(batch: DataProto, tokenizer) -> DataProto:
    """
    [Step 1: Data Cleaning]
    Batch In -> Batch Out
    
    Functionality:
    1. Logical Truncation: Modifies the mask to zero out content after the first <|im_end|>.
    2. Physical Padding: Replaces the masked-out tokens in `input_ids` and `responses` 
       with `pad_token_id`. This is CRITICAL for multi-modal models to prevent 
       the vision encoder from extracting features from garbage/truncated tokens.
    
    Args:
        batch: Original DataProto
        tokenizer: Used to retrieve im_end_id and pad_token_id
    
    Returns:
        DataProto: A deep-copied, sanitized Batch.
    """
    # 1. Deep copy to prevent side effects on the original training data
    new_batch = copy.deepcopy(batch)
    device = new_batch.batch.device
    
    # 2. Retrieve key Token IDs
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
         im_end_id = tokenizer.eos_token_id
    else:
         im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    else:
        # Fallback: if no pad_token_id, use im_end or 0
        pad_token_id = im_end_id 
        
    if isinstance(im_end_id, list): im_end_id = im_end_id[0]
    if isinstance(pad_token_id, list): pad_token_id = pad_token_id[0]

    # 3. Extract data
    responses = new_batch.batch["responses"]
    input_ids = new_batch.batch["input_ids"]
    B, R_len = responses.shape
    Seq_len = input_ids.shape[1]
    P_len = Seq_len - R_len # Length of the Prompt

    # 4. Vectorized search for truncation points
    is_end = (responses == im_end_id)
    end_indices = is_end.float().argmax(dim=1) 
    has_end = is_end.sum(dim=1) > 0
    
    # Calculate valid length (including im_end)
    valid_lens = torch.where(has_end, end_indices + 1, torch.tensor(R_len, device=device))
    
    # 5. Generate Tail Mask
    col_indices = torch.arange(R_len, device=device).unsqueeze(0) 
    cutoff = valid_lens.unsqueeze(1)
    
    # Mask: 1 for valid tokens, 0 for truncated tokens
    tail_mask = (col_indices < cutoff).long() # [B, R]
    
    # 6. [CRITICAL UPGRADE] Physical Padding
    # Force replace masked tokens with Pad ID.
    # This prevents "Image features and tokens do not match" errors in Qwen2-VL.
    
    # a. Modify Responses Tensor
    pad_mask = (tail_mask == 0)
    new_batch.batch["responses"].masked_fill_(pad_mask, pad_token_id)
    
    # b. Modify Input IDs Tensor (Only the response part)
    response_part_input_ids = new_batch.batch["input_ids"][:, P_len:]
    response_part_input_ids.masked_fill_(pad_mask, pad_token_id)
    new_batch.batch["input_ids"][:, P_len:] = response_part_input_ids

    # 7. Update Attention Masks
    if "attention_mask" in new_batch.batch:
        new_batch.batch["attention_mask"][:, -R_len:] *= tail_mask
    if "response_mask" in new_batch.batch:
         new_batch.batch["response_mask"] *= tail_mask
    if "loss_mask" in new_batch.batch:
         new_batch.batch["loss_mask"] *= tail_mask

    return new_batch


# =========================================================================
# 2. Compute MI (Robust Multi-modal & OOM Optimized)
# =========================================================================
def compute_mi(batch: DataProto, actor_wg) -> Dict[str, float]:
    """
    [Step 2] Robust Mutual Information Calculation
    
    Features:
    - OOM Prevention: Uses Chunking (iterating by Prompt). Batch size remains M.
    - Multi-modal Fix: Physically replicates image objects using Python lists.
    - Error Handling: Includes try-except blocks to skip failed batches without crashing.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- A. Basic Information Extraction ---
    original_responses = batch.batch["responses"]   # [M, R] (Already physically padded)
    original_input_ids = batch.batch["input_ids"]   # [M, P+R]
    original_pos_ids = batch.batch.get("position_ids", None)
    
    # Extract Multi-modal inputs
    # Note: In 'verl', multi_modal_inputs can be np.ndarray(dtype=object) or list
    multi_modal_inputs = batch.non_tensor_batch.get("multi_modal_inputs", None)
    has_multi_modal = multi_modal_inputs is not None
    
    M_responses, R_len = original_responses.shape
    P_len = original_input_ids.shape[1] - R_len 
    
    # Get the cleaned Response Mask
    clean_response_mask = batch.batch["attention_mask"][:, -R_len:] # [M, R]
    
    # Group Info
    group_ids = batch.non_tensor_batch.get("group_idx") or batch.non_tensor_batch.get("uid")
    if group_ids is None: group_ids = np.arange(M_responses)
    unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
    N_prompts = len(unique_groups)
    
    if N_prompts < 2: return {}

    # Check for Complex Position IDs (e.g., Qwen2-VL has >2 dims)
    is_complex_pos = (original_pos_ids is not None) and (original_pos_ids.ndim > 2)

    # --- B. Prepare Result Matrix ---
    # matrix_log_probs[j, i] = LogProb(Response i | Prompt j)
    matrix_log_probs = torch.zeros((N_prompts, M_responses), device=device, dtype=torch.float32)

    # --- C. Pre-fetch Prompt Data ---
    prompt_indices = [] 
    for gid in unique_groups:
        idx = np.where(group_ids == gid)[0][0]
        prompt_indices.append(idx)
    
    unique_prompt_input_ids = original_input_ids[prompt_indices, :P_len]
    
    unique_prompt_pos_ids = None
    if is_complex_pos:
        unique_prompt_pos_ids = original_pos_ids[prompt_indices, ..., :P_len]

    # --- D. Chunking Loop (Iterate by Target Prompt) ---
    for j in range(N_prompts):
        
        # 1. Expand Current Prompt (1 -> M)
        curr_prompt_ids = unique_prompt_input_ids[j:j+1].expand(M_responses, -1)
        cross_input_ids = torch.cat([curr_prompt_ids, original_responses], dim=1)
        
        # 2. Prepare Masks
        prompt_mask = (curr_prompt_ids != 0).long()
        cross_attention_mask = torch.cat([prompt_mask, clean_response_mask], dim=1)
        
        # 3. Handle Position IDs
        if is_complex_pos:
            # === Anchor Shifting for Qwen2-VL ===
            # Preserve spatial structure of response, shift based on prompt length diff
            curr_prompt_pos = unique_prompt_pos_ids[j:j+1].expand(M_responses, -1, -1)
            new_anchor = curr_prompt_pos[..., -1:] 
            old_anchors = original_pos_ids[..., :P_len][..., -1:]
            shift = new_anchor - old_anchors
            
            original_response_pos = original_pos_ids[..., -R_len:]
            curr_response_pos = original_response_pos + shift
            cross_position_ids = torch.cat([curr_prompt_pos, curr_response_pos], dim=-1)
        else:
            # === Simple Regeneration ===
            cross_position_ids = torch.cumsum(cross_attention_mask, dim=1) - 1
            cross_position_ids.masked_fill_(cross_attention_mask == 0, 0)
            
        # 4. [CRITICAL FIX] Handle Multi-Modal Inputs
        cross_non_tensor_batch = {}
        if has_multi_modal:
            # Get the single image data object for the current prompt
            src_idx = prompt_indices[j]
            
            # Ensure we get the raw Python object (dict/list), not a numpy wrapper
            if isinstance(multi_modal_inputs, np.ndarray):
                single_image_data = multi_modal_inputs[src_idx]
            else:
                single_image_data = multi_modal_inputs[src_idx]
            
            # Construct List: Replicate the SAME image M times
            # Using Python list multiplication ensures correct object reference copying
            cross_imgs_list = [single_image_data] * M_responses
            
            # [Assertion 1] Check Alignment
            if len(cross_imgs_list) != M_responses:
                logger.error(f"[MI Check] Image list len {len(cross_imgs_list)} != Batch size {M_responses}")
                # Force slice if mismatch occurs (Defensive programming)
                cross_imgs_list = cross_imgs_list[:M_responses]
            
            # Convert back to numpy object array for Verl compatibility
            cross_imgs = np.array(cross_imgs_list, dtype=object)
            
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
            meta_info=batch.meta_info 
        )
        
        # 6. Compute LogProb with Error Handling
        try:
            log_probs_raw, _ = actor_wg.compute_log_prob(cross_batch)
        except ValueError as e:
            # Catch "features and tokens do not match" errors
            # Log error and skip this prompt row (leave as 0s) to prevent crash
            logger.error(f"[MI Error] Failed at prompt {j}: {e}")
            continue
        
        if log_probs_raw.shape[1] != R_len:
            log_probs_raw = log_probs_raw[:, -R_len:]
            
        # 7. Fill Result Matrix
        valid_log_probs = log_probs_raw * clean_response_mask 
        seq_log_probs = valid_log_probs.sum(dim=-1)           
        matrix_log_probs[j, :] = seq_log_probs

    # --- E. Compute MI Metrics ---
    # Transpose to [M, N] -> (Response i, Prompt j)
    matrix_log_probs = matrix_log_probs.t() 
    
    # Matched Score: Diagonal elements
    matched = matrix_log_probs[torch.arange(M_responses), group_indices]
    
    # Marginal Score: LogSumExp over Prompts
    # Note: If some rows failed (0s), this is an approximation. 
    marginal = torch.logsumexp(matrix_log_probs, dim=1) - math.log(N_prompts)
    
    mi_per_sample = matched - marginal
    
    # --- F. Aggregate Results ---
    group_mi_results = {}
    mi_numpy = mi_per_sample.detach().cpu().numpy()
    
    group_mi_results["collapse/mi_global"] = float(np.mean(mi_numpy))
    
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
    1. get_first_turn_batch: Clean batch, apply physical padding (O(M)).
    2. compute_mi: Compute MI using chunked cross-scoring (O(M*N)).
    """
    cleaned_batch = get_first_turn_batch(batch, tokenizer)
    mi_metrics = compute_mi(cleaned_batch, actor_wg)
    return mi_metrics


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
    if batch.non_tensor_batch.get("multi_modal_inputs") is not None:
        logger.warning("Skipping collapse/mi for multi-modal batch to avoid image token/feature mismatch.")
        return {}
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
