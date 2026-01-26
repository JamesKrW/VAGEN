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
# Debug Helpers
# =========================================================================
def _log_sample_debug(batch: DataProto, tokenizer, prefix: str) -> None:
    responses = batch.batch["responses"]
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch.get("attention_mask", None)
    pos_ids = batch.batch.get("position_ids", None)
    multi_modal_inputs = batch.non_tensor_batch.get("multi_modal_inputs", None)
    if responses.numel() == 0:
        logger.info("%s empty_batch", prefix)
        return

    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        im_end_id = tokenizer.eos_token_id
    else:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    try:
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    except Exception:
        im_start_id = None
    if isinstance(im_start_id, list):
        im_start_id = im_start_id[0] if im_start_id else None
    if isinstance(im_end_id, list):
        im_end_id = im_end_id[0]

    idx = 0
    resp = responses[idx]
    attn_resp = attention_mask[idx, -responses.shape[1] :] if attention_mask is not None else None
    valid = attn_resp > 0 if attn_resp is not None else None
    if valid is not None:
        starts = torch.nonzero((resp == im_start_id) & valid, as_tuple=False).flatten() if im_start_id is not None else torch.empty(0, device=resp.device)
        ends = torch.nonzero((resp == im_end_id) & valid, as_tuple=False).flatten()
    else:
        starts = torch.nonzero(resp == im_start_id, as_tuple=False).flatten() if im_start_id is not None else torch.empty(0, device=resp.device)
        ends = torch.nonzero(resp == im_end_id, as_tuple=False).flatten()

    mm_info = "none"
    if multi_modal_inputs is not None:
        try:
            mm = multi_modal_inputs[idx] if isinstance(multi_modal_inputs, np.ndarray) else multi_modal_inputs[idx]
            if isinstance(mm, dict):
                grid = mm.get("image_grid_thw")
                pv = mm.get("pixel_values")
                grid_shape = tuple(grid.shape) if torch.is_tensor(grid) else (grid.shape if hasattr(grid, "shape") else None)
                if torch.is_tensor(pv):
                    pv_len = int(pv.shape[0])
                elif isinstance(pv, (list, tuple, np.ndarray)):
                    pv_len = len(pv)
                else:
                    pv_len = None
                mm_info = f"grid={grid_shape} pv_len={pv_len}"
            else:
                mm_info = f"type={type(mm)}"
        except Exception as e:
            mm_info = f"err={e}"

    logger.info(
        "%s B=%d P_len=%d R_len=%d input_len=%d pos_shape=%s starts=%d ends=%d first_start=%s first_end=%s mm=%s",
        prefix,
        responses.shape[0],
        input_ids.shape[1] - responses.shape[1],
        responses.shape[1],
        input_ids.shape[1],
        str(None if pos_ids is None else tuple(pos_ids.shape)),
        int(starts.numel()),
        int(ends.numel()),
        str(int(starts[0].item())) if starts.numel() > 0 else "None",
        str(int(ends[0].item())) if ends.numel() > 0 else "None",
        mm_info,
    )


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
    prompt_lens = []
    response_lens = []

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
        prompt_lens.append(int(prompt_seq.shape[0]))
        response_lens.append(int(suffix.shape[0]))

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

            new_pos_prompts.append(pos_prompt_seq)
            new_pos_responses.append(pos_suffix)

    new_P_len = max(prompt_lens) if prompt_lens else P_len
    new_R_len = max(response_lens) if response_lens else R_len

    padded_prompts = []
    padded_responses = []
    for prompt_seq, resp_seq in zip(new_prompts, new_responses):
        if prompt_seq.shape[0] < new_P_len:
            pad = torch.full(
                (new_P_len - prompt_seq.shape[0],),
                pad_token_id,
                dtype=prompt_seq.dtype,
                device=prompt_seq.device,
            )
            prompt_seq = torch.cat([pad, prompt_seq], dim=0)
        if resp_seq.shape[0] < new_R_len:
            pad = torch.full(
                (new_R_len - resp_seq.shape[0],),
                pad_token_id,
                dtype=resp_seq.dtype,
                device=resp_seq.device,
            )
            resp_seq = torch.cat([resp_seq, pad], dim=0)
        padded_prompts.append(prompt_seq)
        padded_responses.append(resp_seq)

    new_prompts = torch.stack(padded_prompts, dim=0)
    new_responses = torch.stack(padded_responses, dim=0)
    new_input_ids = torch.cat([new_prompts, new_responses], dim=1)

    prompt_mask = (new_prompts != pad_token_id).long()
    response_mask = (new_responses != pad_token_id).long()
    new_attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

    new_batch.batch["responses"] = new_responses
    new_batch.batch["input_ids"] = new_input_ids
    new_batch.batch["attention_mask"] = new_attention_mask
    new_batch.batch["response_mask"] = response_mask
    if "loss_mask" in new_batch.batch:
        new_batch.batch["loss_mask"] = response_mask.to(new_batch.batch["loss_mask"].dtype)

    if pos_ids is not None:
        padded_pos_prompts = []
        padded_pos_responses = []
        for pos_prompt_seq, pos_resp_seq in zip(new_pos_prompts, new_pos_responses):
            if pos_prompt_seq.shape[-1] < new_P_len:
                pad_shape = list(pos_prompt_seq.shape)
                pad_shape[-1] = new_P_len - pos_prompt_seq.shape[-1]
                pad = torch.zeros(pad_shape, dtype=pos_prompt_seq.dtype, device=pos_prompt_seq.device)
                pos_prompt_seq = torch.cat([pad, pos_prompt_seq], dim=-1)
            if pos_resp_seq.shape[-1] < new_R_len:
                pad_shape = list(pos_resp_seq.shape)
                pad_shape[-1] = new_R_len - pos_resp_seq.shape[-1]
                pad = torch.zeros(pad_shape, dtype=pos_resp_seq.dtype, device=pos_resp_seq.device)
                pos_resp_seq = torch.cat([pos_resp_seq, pad], dim=-1)
            padded_pos_prompts.append(pos_prompt_seq)
            padded_pos_responses.append(pos_resp_seq)
        new_pos_prompts = torch.stack(padded_pos_prompts, dim=0)
        new_pos_responses = torch.stack(padded_pos_responses, dim=0)
        new_batch.batch["position_ids"] = torch.cat([new_pos_prompts, new_pos_responses], dim=-1)

    logger.info(
        "[MI Debug][Multi][Rebuild] idx=%d P_len=%d R_len=%d B=%d im_start_id=%s",
        idx,
        new_P_len,
        new_R_len,
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
    try:
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    except Exception:
        im_start_id = None
    if isinstance(im_start_id, list):
        im_start_id = im_start_id[0] if im_start_id else None

    responses = batch.batch["responses"]
    attention_mask = batch.batch.get("attention_mask", None)
    if attention_mask is not None:
        response_mask = attention_mask[:, -responses.shape[1]:] > 0
        valid_resp = response_mask
    else:
        valid_resp = None

    start_counts = []
    end_positions_first = []
    start_positions_first = []
    for i in range(responses.shape[0]):
        resp = responses[i]
        if valid_resp is not None:
            valid = valid_resp[i] > 0
            starts = torch.nonzero((resp == im_start_id) & valid, as_tuple=False).flatten() if im_start_id is not None else torch.empty(0, device=resp.device)
            ends = torch.nonzero((resp == im_end_id) & valid, as_tuple=False).flatten()
        else:
            starts = torch.nonzero(resp == im_start_id, as_tuple=False).flatten() if im_start_id is not None else torch.empty(0, device=resp.device)
            ends = torch.nonzero(resp == im_end_id, as_tuple=False).flatten()
        start_counts.append(int(starts.numel()))
        start_positions_first.append(int(starts[0].item()) if starts.numel() > 0 else None)
        end_positions_first.append(int(ends[0].item()) if ends.numel() > 0 else None)

    turn_counts = []
    for sc, first_end, first_start in zip(start_counts, end_positions_first, start_positions_first):
        if first_end is not None and (first_start is None or first_end < first_start):
            turn_counts.append(sc + 1)
        else:
            turn_counts.append(sc)
    if turn_counts:
        logger.info(
            "[MI Debug][Multi] turn_counts min=%d max=%d unique=%d im_start_id=%s im_end_id=%s",
            int(min(turn_counts)),
            int(max(turn_counts)),
            len(set(int(c) for c in turn_counts)),
            str(im_start_id),
            str(im_end_id),
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
        _log_sample_debug(cleaned, tokenizer, f"[MI Debug][Multi][Sample] t={turn}")
        try:
            mi_metrics = compute_mi(cleaned, actor_wg, tokenizer=tokenizer, processor=processor)
        except Exception as e:
            logger.error("[MI Error][Multi] compute_mi failed t=%d: %s", turn, str(e))
            _log_sample_debug(cleaned, tokenizer, f"[MI Debug][Multi][Sample][Fail] t={turn}")
            continue
        for k, v in mi_metrics.items():
            if k == "collapse/mi_global":
                results[f"collapse/mi_global_t{turn}"] = v
            else:
                results[f"{str(k)}/t{turn}"] = v

    return results
