import copy
import math
import numpy as np
import torch
import logging
from typing import Dict, Optional, List, Any, Tuple
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
    logger.info(
        "[MI Debug][Step1] B=%d P_len=%d R_len=%d Seq_len=%d im_end_id=%s pad_token_id=%s",
        B, P_len, R_len, Seq_len, str(im_end_id), str(pad_token_id)
    )

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
    if B > 0:
        num_with_end = int(has_end.sum().item())
        logger.info(
            "[MI Debug][Step1] has_end=%d/%d valid_len_min=%d valid_len_max=%d",
            num_with_end,
            B,
            int(valid_lens.min().item()),
            int(valid_lens.max().item()),
        )
    
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
    if B > 0:
        logger.info("[MI Debug][Step1] padded_tokens=%d", int(pad_mask.sum().item()))

    # 7. Update Attention Masks
    if "attention_mask" in new_batch.batch:
        new_batch.batch["attention_mask"][:, -R_len:] *= tail_mask
    if "response_mask" in new_batch.batch:
         new_batch.batch["response_mask"] *= tail_mask
    if "loss_mask" in new_batch.batch:
         new_batch.batch["loss_mask"] *= tail_mask

    return new_batch


# =========================================================================
# 1.5 Keep One Sample Per Group
# =========================================================================
def get_unique_group_idx_dp(batch: DataProto) -> DataProto:
    """
    Keep only the first sample for each group_idx (or uid) in order.
    """
    group_ids = batch.non_tensor_batch.get("group_idx") or batch.non_tensor_batch.get("uid")
    if group_ids is None:
        return batch

    if isinstance(group_ids, np.ndarray):
        group_list = group_ids.tolist()
    else:
        group_list = list(group_ids)

    seen = set()
    keep_indices = []
    for i, gid in enumerate(group_list):
        try:
            key = gid
            if key in seen:
                continue
            seen.add(key)
        except TypeError:
            key = str(gid)
            if key in seen:
                continue
            seen.add(key)
        keep_indices.append(i)

    if len(keep_indices) == len(group_list):
        return batch
    return batch[keep_indices]


# =========================================================================
# 2. Compute MI (Robust Multi-modal & OOM Optimized)
# =========================================================================
def _get_spatial_merge_size(actor_wg, processor=None) -> Optional[int]:
    if processor is not None:
        image_processor = getattr(processor, "image_processor", None)
        merge_size = getattr(image_processor, "merge_size", None)
        if merge_size is not None:
            return int(merge_size)
    for attr in ("actor_module", "actor_model", "model", "module"):
        module = getattr(actor_wg, attr, None)
        if module is None:
            continue
        config = getattr(module, "config", None)
        if config is None:
            continue
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            merge_size = getattr(vision_config, "spatial_merge_size", None)
            if merge_size is not None:
                return int(merge_size)
        merge_size = getattr(config, "spatial_merge_size", None)
        if merge_size is not None:
            return int(merge_size)
    return None


def _get_image_token_id(tokenizer, processor=None) -> Optional[int]:
    if processor is not None:
        processor_tokenizer = getattr(processor, "tokenizer", None)
        if processor_tokenizer is not None:
            image_token_id = getattr(processor_tokenizer, "image_token_id", None)
            if image_token_id is not None:
                return int(image_token_id)
            try:
                image_token_id = processor_tokenizer.convert_tokens_to_ids("<|image_pad|>")
                if isinstance(image_token_id, list):
                    image_token_id = image_token_id[0] if image_token_id else None
                return int(image_token_id) if image_token_id is not None else None
            except Exception:
                pass
    if tokenizer is None:
        return None
    image_token_id = getattr(tokenizer, "image_token_id", None)
    if image_token_id is None:
        try:
            image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        except Exception:
            image_token_id = None
    if isinstance(image_token_id, list):
        image_token_id = image_token_id[0] if image_token_id else None
    return int(image_token_id) if image_token_id is not None else None


def _count_image_tokens(input_ids: torch.Tensor, image_token_id: Optional[int]) -> Optional[int]:
    if image_token_id is None:
        return None
    return int((input_ids == image_token_id).sum().item())


def _estimate_image_tokens_from_grid(
    image_grid_thw: torch.Tensor, merge_size: int
) -> Tuple[int, List[int]]:
    grid_tokens: List[int] = []
    for row in image_grid_thw:
        t = int(row[0].item())
        h = int(row[1].item())
        w = int(row[2].item())
        grid_tokens.append(t * (h // merge_size) * (w // merge_size))
    return sum(grid_tokens), grid_tokens


def _estimate_image_patches_from_grid(image_grid_thw: torch.Tensor) -> List[int]:
    grid_patches: List[int] = []
    for row in image_grid_thw:
        t = int(row[0].item())
        h = int(row[1].item())
        w = int(row[2].item())
        grid_patches.append(t * h * w)
    return grid_patches


def _align_multimodal_tokens_and_features(
    cross_input_ids: torch.Tensor,
    cross_attention_mask: torch.Tensor,
    single_image_data: dict,
    tokenizer,
    merge_size: Optional[int],
    processor,
    pad_token_id: int,
    logger,
) -> dict:
    """
    Validate and align image tokens with image features.
    Steps:
      1) Validate counts.
      2) Truncate features if features > tokens.
      3) Pad extra image tokens if tokens > features.
      4) Re-validate and log.
    """
    if merge_size is None or single_image_data is None:
        return single_image_data
    image_token_id = _get_image_token_id(tokenizer, processor=processor)
    if image_token_id is None:
        return single_image_data
    image_grid_thw = single_image_data.get("image_grid_thw")
    if image_grid_thw is None:
        return single_image_data
    if not torch.is_tensor(image_grid_thw):
        image_grid_thw = torch.as_tensor(image_grid_thw)

    feature_tokens, _ = _estimate_image_tokens_from_grid(image_grid_thw, merge_size)
    token_counts = (cross_input_ids == image_token_id).sum(dim=1).tolist()
    max_tokens = int(max(token_counts)) if token_counts else 0

    # Step 2: Truncate features if needed
    if feature_tokens > max_tokens and max_tokens > 0:
        ref_idx = int(token_counts.index(max_tokens))
        single_image_data = _truncate_multi_modal_inputs_for_tokens(
            single_image_data,
            cross_input_ids[ref_idx],
            tokenizer,
            merge_size,
            processor,
            logger,
            pad_token_id,
        )
        image_grid_thw = single_image_data.get("image_grid_thw")
        if image_grid_thw is not None and not torch.is_tensor(image_grid_thw):
            image_grid_thw = torch.as_tensor(image_grid_thw)
        if image_grid_thw is not None:
            feature_tokens, _ = _estimate_image_tokens_from_grid(image_grid_thw, merge_size)

    # Step 3: Pad extra image tokens if needed
    if feature_tokens > 0:
        for i, n_tokens in enumerate(token_counts):
            if n_tokens <= feature_tokens:
                continue
            positions = torch.nonzero(cross_input_ids[i] == image_token_id, as_tuple=False).flatten()
            extra = positions[feature_tokens:]
            if extra.numel() > 0:
                cross_input_ids[i, extra] = pad_token_id
                cross_attention_mask[i, extra] = 0

    # Step 4: Re-validate
    new_counts = (cross_input_ids == image_token_id).sum(dim=1)
    if (new_counts > feature_tokens).any():
        logger.warning(
            "[MI Debug][MM] mismatch after align: feature_tokens=%d max_tokens=%d",
            int(feature_tokens),
            int(new_counts.max().item()),
        )
    return single_image_data


def _truncate_multi_modal_inputs_for_tokens(
    single_image_data: dict,
    input_ids: torch.Tensor,
    tokenizer,
    merge_size: Optional[int],
    processor,
    logger,
    pad_token_id: int,
) -> dict:
    if merge_size is None:
        logger.warning("[MI Debug][MM] merge_size missing; skip multi-modal alignment.")
        return single_image_data
    image_token_id = _get_image_token_id(tokenizer, processor=processor)
    n_image_tokens = _count_image_tokens(input_ids, image_token_id)
    if n_image_tokens is None:
        logger.warning("[MI Debug][MM] image_token_id missing; skip multi-modal alignment.")
        return single_image_data
    image_grid_thw = single_image_data.get("image_grid_thw")
    if image_grid_thw is None:
        logger.warning("[MI Debug][MM] image_grid_thw missing; skip multi-modal alignment.")
        return single_image_data
    if not torch.is_tensor(image_grid_thw):
        image_grid_thw = torch.as_tensor(image_grid_thw)
    if n_image_tokens == 0:
        logger.warning("[MI Debug][MM] input_ids has 0 image tokens; dropping image inputs.")
        new_data = dict(single_image_data)
        for key in ("pixel_values", "image_grid_thw"):
            new_data.pop(key, None)
        return new_data
    total_tokens, grid_tokens = _estimate_image_tokens_from_grid(image_grid_thw, merge_size)
    if total_tokens == n_image_tokens:
        return single_image_data

    logger.warning(
        "[MI Debug][MM] token mismatch: input_ids=%d grid_est=%d images=%d",
        n_image_tokens,
        total_tokens,
        int(image_grid_thw.shape[0]),
    )
    if total_tokens > n_image_tokens and n_image_tokens > 0:
        cumulative = 0
        keep_images = 0
        for count in grid_tokens:
            if cumulative + count > n_image_tokens:
                break
            cumulative += count
            keep_images += 1
        if keep_images == 0:
            keep_images = 1
            cumulative = grid_tokens[0]

        new_data = {}
        num_images = int(image_grid_thw.shape[0])
        patches_per_image = _estimate_image_patches_from_grid(image_grid_thw)
        keep_patches = int(sum(patches_per_image[:keep_images]))
        for key, val in single_image_data.items():
            if torch.is_tensor(val):
                if val.shape[0] == num_images:
                    new_data[key] = val[:keep_images]
                elif key == "pixel_values" and val.shape[0] >= keep_patches:
                    new_data[key] = val[:keep_patches]
                else:
                    new_data[key] = val
            elif isinstance(val, np.ndarray):
                if val.shape[0] == num_images:
                    new_data[key] = val[:keep_images]
                elif key == "pixel_values" and val.shape[0] >= keep_patches:
                    new_data[key] = val[:keep_patches]
                else:
                    new_data[key] = val
            elif isinstance(val, (list, tuple)):
                if len(val) == num_images:
                    new_data[key] = val[:keep_images]
                elif key == "pixel_values" and len(val) >= keep_patches:
                    new_data[key] = val[:keep_patches]
                else:
                    new_data[key] = val
            else:
                new_data[key] = val

        new_grid = new_data.get("image_grid_thw")
        if new_grid is not None:
            if not torch.is_tensor(new_grid):
                new_grid = torch.as_tensor(new_grid)
            new_total, _ = _estimate_image_tokens_from_grid(new_grid, merge_size)
            if new_total != n_image_tokens:
                logger.error(
                    "[MI Debug][MM] mismatch after truncate: input_ids=%d grid_est=%d keep_images=%d",
                    n_image_tokens,
                    new_total,
                    keep_images,
                )
                assert new_total == n_image_tokens, "image token mismatch after truncation"
        if "pixel_values" in new_data and new_grid is not None:
            pixel_values = new_data["pixel_values"]
            if torch.is_tensor(pixel_values):
                pv_len = int(pixel_values.shape[0])
            elif isinstance(pixel_values, np.ndarray):
                pv_len = int(pixel_values.shape[0])
            elif isinstance(pixel_values, (list, tuple)):
                pv_len = len(pixel_values)
            else:
                pv_len = None
            if pv_len is not None:
                patches_per_image = _estimate_image_patches_from_grid(new_grid)
                expected_patches = int(sum(patches_per_image))
                if pv_len != expected_patches:
                    logger.warning(
                        "[MI Debug][MM] pixel_values len != grid patches: pv_len=%d expected=%d",
                        pv_len,
                        expected_patches,
                    )
        logger.warning(
            "[MI Debug][MM] truncated images to keep_images=%d grid_est=%d target=%d keep_patches=%d",
            keep_images,
            cumulative,
            n_image_tokens,
            keep_patches,
        )
        return new_data

    if total_tokens < n_image_tokens:
        # Reduce image token count by padding extra image tokens in input_ids.
        image_token_id = _get_image_token_id(tokenizer, processor=processor)
        if image_token_id is None:
            logger.warning("[MI Debug][MM] image_token_id missing; cannot pad extra image tokens.")
            return single_image_data
        image_positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten()
        if image_positions.numel() > total_tokens:
            extra = image_positions[total_tokens:]
            input_ids[extra] = pad_token_id
            logger.warning(
                "[MI Debug][MM] padded extra image tokens: tokens=%d features=%d padded=%d",
                n_image_tokens,
                total_tokens,
                int(extra.numel()),
            )
        return single_image_data

    logger.error(
        "[MI Debug][MM] token mismatch cannot be fixed by truncation: input_ids=%d grid_est=%d",
        n_image_tokens,
        total_tokens,
    )
    assert total_tokens == n_image_tokens, "image token mismatch without truncation"
    return single_image_data


def compute_mi(batch: DataProto, actor_wg, tokenizer=None, processor=None) -> Dict[str, float]:
    """
    [Step 2] Robust Mutual Information Calculation
    
    Features:
    - OOM Prevention: Uses Chunking (iterating by Prompt). Batch size remains M.
    - Multi-modal Fix: Physically replicates image objects using Python lists.
    - Error Handling: Includes try-except blocks to skip failed batches without crashing.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    merge_size = _get_spatial_merge_size(actor_wg, processor=processor)
    pad_token_id = 0
    if tokenizer is not None and hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0] if pad_token_id else 0
    
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
    
    if N_prompts < 2:
        logger.info("[MI Debug][Step2] N_prompts=%d < 2, skipping MI", N_prompts)
        return {}

    # Check for Complex Position IDs (e.g., Qwen2-VL has >2 dims)
    is_complex_pos = (original_pos_ids is not None) and (original_pos_ids.ndim > 2)
    logger.info(
        "[MI Debug][Step2] M=%d P_len=%d R_len=%d N_prompts=%d has_multi_modal=%s pos_ids_shape=%s",
        M_responses,
        P_len,
        R_len,
        N_prompts,
        str(has_multi_modal),
        str(None if original_pos_ids is None else tuple(original_pos_ids.shape)),
    )

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
        # Ensure masked response tokens (incl. image tokens) do not count as image tokens
        image_token_id = _get_image_token_id(tokenizer, processor=processor)
        if image_token_id is not None:
            masked_positions = clean_response_mask == 0
            if masked_positions.any():
                response_slice = cross_input_ids[:, -R_len:]
                image_masked = (response_slice == image_token_id) & masked_positions
                if image_masked.any():
                    response_slice = response_slice.masked_fill(image_masked, pad_token_id)
                    cross_input_ids[:, -R_len:] = response_slice
        
        # 2. Prepare Masks
        prompt_mask = (curr_prompt_ids != pad_token_id).long()
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
            
            if isinstance(single_image_data, dict):
                single_image_data = _truncate_multi_modal_inputs_for_tokens(
                    single_image_data,
                    curr_prompt_ids[0],
                    tokenizer,
                    merge_size,
                    processor,
                    logger,
                    pad_token_id,
                )

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

            # Validate and align tokens/features before compute_log_prob
            single_image_data = _align_multimodal_tokens_and_features(
                cross_input_ids,
                cross_attention_mask,
                single_image_data,
                tokenizer,
                merge_size,
                processor,
                pad_token_id,
                logger,
            )
            if isinstance(single_image_data, dict):
                cross_non_tensor_batch["multi_modal_inputs"] = np.array(
                    [single_image_data] * M_responses, dtype=object
                )
        if j == 0:
            logger.info(
                "[MI Debug][Step2] j=%d cross_input_ids=%s cross_attention_mask=%s cross_position_ids=%s multi_modal_type=%s",
                j,
                str(tuple(cross_input_ids.shape)),
                str(tuple(cross_attention_mask.shape)),
                str(tuple(cross_position_ids.shape)),
                str(type(multi_modal_inputs)),
            )

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
            log_prob_out = actor_wg.compute_log_prob(cross_batch)
        except ValueError as e:
            # Catch "features and tokens do not match" errors
            # Log error and skip this prompt row (leave as 0s) to prevent crash
            logger.error(
                "[MI Error] Failed at prompt %d: %s | cross_input_ids=%s cross_attention_mask=%s cross_position_ids=%s",
                j,
                str(e),
                str(tuple(cross_input_ids.shape)),
                str(tuple(cross_attention_mask.shape)),
                str(tuple(cross_position_ids.shape)),
            )
            continue
        except RuntimeError as e:
            pv_len = None
            grid_shape = None
            if has_multi_modal and isinstance(single_image_data, dict):
                grid_thw = single_image_data.get("image_grid_thw")
                if grid_thw is not None:
                    if torch.is_tensor(grid_thw):
                        grid_shape = tuple(grid_thw.shape)
                    elif isinstance(grid_thw, np.ndarray):
                        grid_shape = grid_thw.shape
                    else:
                        try:
                            grid_shape = (len(grid_thw),)
                        except Exception:
                            grid_shape = None
                pixel_values = single_image_data.get("pixel_values")
                if pixel_values is not None:
                    if torch.is_tensor(pixel_values):
                        pv_len = int(pixel_values.shape[0])
                    elif isinstance(pixel_values, np.ndarray):
                        pv_len = int(pixel_values.shape[0])
                    elif isinstance(pixel_values, (list, tuple)):
                        pv_len = len(pixel_values)
            logger.error(
                "[MI Error] RuntimeError at prompt %d: %s | grid_thw=%s pixel_values_len=%s",
                j,
                str(e),
                str(grid_shape),
                str(pv_len),
            )
            continue

        if isinstance(log_prob_out, tuple):
            log_probs_raw = log_prob_out[0]
        elif isinstance(log_prob_out, DataProto):
            if "old_log_probs" in log_prob_out.batch:
                log_probs_raw = log_prob_out.batch["old_log_probs"]
            elif "log_probs" in log_prob_out.batch:
                log_probs_raw = log_prob_out.batch["log_probs"]
            else:
                logger.error(
                    "[MI Error] compute_log_prob returned DataProto without log probs at prompt %d",
                    j,
                )
                continue
        elif torch.is_tensor(log_prob_out):
            log_probs_raw = log_prob_out
        else:
            logger.error(
                "[MI Error] compute_log_prob returned unsupported type %s at prompt %d",
                str(type(log_prob_out)),
                j,
            )
            continue

        if log_probs_raw.device != clean_response_mask.device:
            log_probs_raw = log_probs_raw.to(clean_response_mask.device)
        
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
            group_mi_results[gid] = float(np.mean(mi_numpy[mask]))
            
    return group_mi_results


# =========================================================================
# 3. Entry Point
# =========================================================================
def compute_group_mi_first_turn(batch: DataProto, actor_wg, tokenizer, processor=None) -> Dict[str, float]:
    """
    Main Entry Point.
    
    Pipeline:
    1. Clean Batch: Logically truncate to first turn (O(M)).
    2. Compute MI: Perform chunked cross-scoring (O(M*N)).
    """
    batch = get_unique_group_idx_dp(batch)
    logger.info("[MI Debug][Entry] Starting MI compute for batch size=%d", batch.batch["responses"].shape[0])
    cleaned_batch = get_first_turn_batch(batch, tokenizer)
    mi_metrics = compute_mi(cleaned_batch, actor_wg, tokenizer=tokenizer, processor=processor)
    return mi_metrics
