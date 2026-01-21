import copy
import numpy as np
import torch
import logging
from typing import Dict
from verl import DataProto

from .mutual_inforamtion_single_turn import get_first_turn_batch, compute_mi

# Initialize logger for error reporting
logger = logging.getLogger(__name__)


# =========================================================================
# Multi-turn Prompt/Response Rebuild
# =========================================================================
def build_new_prompt_response(batch: DataProto, idx: int, tokenizer) -> DataProto:
    """
    Move tokens before the idx-th <|im_end|> from response into prompt.
    idx=0 keeps the original prompt/response.
    """
    if idx <= 0:
        return batch

    # Token IDs
    try:
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    except Exception:
        im_start_id = None
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = 0
    if isinstance(im_start_id, list):
        im_start_id = im_start_id[0] if im_start_id else None
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0]
    if im_start_id is None:
        logger.warning("[MI Debug][Multi][Rebuild] im_start_id missing; no-op for idx=%d", idx)
        return batch

    new_batch = copy.deepcopy(batch)
    responses = new_batch.batch["responses"]
    input_ids = new_batch.batch["input_ids"]
    pos_ids = new_batch.batch.get("position_ids", None)
    attention_mask = new_batch.batch.get("attention_mask", None)

    B, R_len = responses.shape
    P_len = input_ids.shape[1] - R_len

    prompt_ids = input_ids[:, :P_len]
    response_mask = None
    if attention_mask is not None:
        response_mask = attention_mask[:, -R_len:]

    new_prompts = []
    new_responses = []
    new_pos_prompts = [] if pos_ids is not None else None
    new_pos_responses = [] if pos_ids is not None else None

    for i in range(B):
        resp = responses[i]
        if response_mask is not None:
            valid = response_mask[i] > 0
            start_positions = torch.nonzero((resp == im_start_id) & valid, as_tuple=False).flatten()
        else:
            start_positions = torch.nonzero(resp == im_start_id, as_tuple=False).flatten()

        if start_positions.numel() < idx:
            cutoff = None
        else:
            cutoff = int(start_positions[idx - 1].item())

        if cutoff is None:
            prefix = resp[:0]
            suffix = resp
        else:
            prefix = resp[: cutoff + 1]
            suffix = resp[cutoff + 1 :]

        prompt_seq = torch.cat([prompt_ids[i], prefix], dim=0)
        if prompt_seq.shape[0] > P_len:
            prompt_seq = prompt_seq[-P_len:]
        elif prompt_seq.shape[0] < P_len:
            pad = torch.zeros(P_len - prompt_seq.shape[0], dtype=prompt_seq.dtype, device=prompt_seq.device)
            prompt_seq = torch.cat([pad, prompt_seq], dim=0)

        if suffix.shape[0] > R_len:
            suffix = suffix[:R_len]
        elif suffix.shape[0] < R_len:
            pad = torch.full(
                (R_len - suffix.shape[0],),
                pad_token_id,
                dtype=suffix.dtype,
                device=suffix.device,
            )
            suffix = torch.cat([suffix, pad], dim=0)

        new_prompts.append(prompt_seq)
        new_responses.append(suffix)

        if pos_ids is not None:
            pos_prompt = pos_ids[i, ..., :P_len]
            pos_resp = pos_ids[i, ..., P_len:]
            if cutoff is None:
                pos_prefix = pos_resp[..., :0]
                pos_suffix = pos_resp
            else:
                pos_prefix = pos_resp[..., : cutoff + 1]
                pos_suffix = pos_resp[..., cutoff + 1 :]

            pos_prompt_seq = torch.cat([pos_prompt, pos_prefix], dim=-1)
            if pos_prompt_seq.shape[-1] > P_len:
                pos_prompt_seq = pos_prompt_seq[..., -P_len:]
            elif pos_prompt_seq.shape[-1] < P_len:
                pad_shape = list(pos_prompt_seq.shape)
                pad_shape[-1] = P_len - pos_prompt_seq.shape[-1]
                pad = torch.zeros(pad_shape, dtype=pos_prompt_seq.dtype, device=pos_prompt_seq.device)
                pos_prompt_seq = torch.cat([pad, pos_prompt_seq], dim=-1)

            if pos_suffix.shape[-1] > R_len:
                pos_suffix = pos_suffix[..., :R_len]
            elif pos_suffix.shape[-1] < R_len:
                pad_shape = list(pos_suffix.shape)
                pad_shape[-1] = R_len - pos_suffix.shape[-1]
                pad = torch.zeros(pad_shape, dtype=pos_suffix.dtype, device=pos_suffix.device)
                pos_suffix = torch.cat([pos_suffix, pad], dim=-1)

            new_pos_prompts.append(pos_prompt_seq)
            new_pos_responses.append(pos_suffix)

    new_prompts = torch.stack(new_prompts, dim=0)
    new_responses = torch.stack(new_responses, dim=0)
    new_input_ids = torch.cat([new_prompts, new_responses], dim=1)

    prompt_mask = (new_prompts != 0).long()
    response_mask = (new_responses != pad_token_id).long()
    new_attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

    new_batch.batch["responses"] = new_responses
    new_batch.batch["input_ids"] = new_input_ids
    new_batch.batch["attention_mask"] = new_attention_mask
    if "response_mask" in new_batch.batch:
        new_batch.batch["response_mask"] = response_mask.to(new_batch.batch["response_mask"].dtype)
    if "loss_mask" in new_batch.batch:
        new_batch.batch["loss_mask"] = response_mask.to(new_batch.batch["loss_mask"].dtype)

    if pos_ids is not None:
        new_pos_prompts = torch.stack(new_pos_prompts, dim=0)
        new_pos_responses = torch.stack(new_pos_responses, dim=0)
        new_batch.batch["position_ids"] = torch.cat([new_pos_prompts, new_pos_responses], dim=-1)

    logger.info(
        "[MI Debug][Multi][Rebuild] idx=%d P_len=%d R_len=%d B=%d im_start_id=%s",
        idx,
        P_len,
        R_len,
        B,
        str(im_start_id),
    )
    return new_batch


# =========================================================================
# Multi-turn MI
# =========================================================================
def compute_group_mi_multi_turn(batch: DataProto, actor_wg, tokenizer, processor=None) -> Dict[str, float]:
    """
    Compute multi-turn MI by aligning turns across groups.
    """
    group_ids = batch.non_tensor_batch.get("group_idx") or batch.non_tensor_batch.get("uid")
    if group_ids is None:
        return {}

    if isinstance(group_ids, np.ndarray):
        group_list = group_ids.tolist()
    else:
        group_list = list(group_ids)
    logger.info("[MI Debug][Multi] total_samples=%d", len(group_list))

    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        im_end_id = tokenizer.eos_token_id
    else:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, list):
        im_end_id = im_end_id[0]

    responses = batch.batch["responses"]
    attention_mask = batch.batch.get("attention_mask", None)
    if attention_mask is not None:
        response_mask = attention_mask[:, -responses.shape[1]:] > 0
        turn_counts = ((responses == im_end_id) & response_mask).sum(dim=1).cpu().tolist()
    else:
        turn_counts = (responses == im_end_id).sum(dim=1).cpu().tolist()
    if turn_counts:
        logger.info(
            "[MI Debug][Multi] turn_counts min=%d max=%d unique=%d",
            int(min(turn_counts)),
            int(max(turn_counts)),
            len(set(int(c) for c in turn_counts)),
        )

    group_order = []
    group_key_by_idx = []
    key_to_gid = {}
    for gid in group_list:
        try:
            hash(gid)
            key = gid
        except TypeError:
            key = str(gid)
        group_key_by_idx.append(key)
        if key not in key_to_gid:
            key_to_gid[key] = gid
            group_order.append(key)

    group_to_counts = {key: set() for key in group_order}
    for key, count in zip(group_key_by_idx, turn_counts):
        group_to_counts[key].add(int(count))

    common_counts = None
    for key in group_order:
        if common_counts is None:
            common_counts = set(group_to_counts[key])
        else:
            common_counts &= group_to_counts[key]
    if not common_counts:
        logger.info("[MI Debug][Multi] No common turn count across groups.")
        return {}

    max_turns = max(common_counts)
    logger.info(
        "[MI Debug][Multi] common_counts=%s using_max=%d",
        sorted(int(c) for c in common_counts),
        max_turns,
    )

    keep_indices = []
    for key in group_order:
        for i, k in enumerate(group_key_by_idx):
            if k == key and turn_counts[i] == max_turns:
                keep_indices.append(i)
                break

    if not keep_indices:
        return {}

    sub_batch = batch[keep_indices]
    logger.info("[MI Debug][Multi] selected_samples=%d", len(keep_indices))
    results: Dict[str, float] = {}
    for turn in range(1, max_turns + 1):
        logger.info("[MI Debug][Multi] computing_turn=%d/%d", turn, max_turns)
        rebuilt = build_new_prompt_response(sub_batch, idx=turn - 1, tokenizer=tokenizer)
        cleaned = get_first_turn_batch(rebuilt, tokenizer)
        mi_metrics = compute_mi(cleaned, actor_wg, tokenizer=tokenizer, processor=processor)
        for k, v in mi_metrics.items():
            if k == "collapse/mi_global":
                results[f"collapse/mi_global_t{turn}"] = v
            else:
                results[f"{str(k)}/t{turn}"] = v

    return results
