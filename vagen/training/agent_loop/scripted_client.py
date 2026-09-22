"""A rollout client that replays scripted responses instead of sampling.

Used to push demonstrations through the exact rollout path (harness, environment, image
tensors, TransferQueue rows): the environment is stepped on the demonstration's actions
and the rows carry the demonstration's text as the "model output". With a constant
advantage the PPO loss on such rows is the masked NLL of the demonstration.
"""
from __future__ import annotations

from typing import Any

from vagen.rollout.client import BackendOutput
from vagen.training.agent_loop.verl_client import VerlClient


class ScriptedClient(VerlClient):
    """``VerlClient`` whose ``generate`` returns the next scripted response."""

    def __init__(self, *args, responses: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self._scripted = list(responses)
        self._next = 0

    async def generate(self, prompt_ids: list[int], **kwargs) -> BackendOutput:
        if self._next >= len(self._scripted):
            raise RuntimeError(
                f"scripted episode asked for response {self._next + 1} but only "
                f"{len(self._scripted)} were provided"
            )
        text = self._scripted[self._next]
        self._next += 1
        token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        # The engine's samples end with the chat end-of-turn token; the training rows
        # must too, or the demonstration teaches an answer that never stops.
        eos = self.tokenizer.eos_token_id
        if eos is not None and (not token_ids or token_ids[-1] != eos):
            token_ids.append(eos)
        return BackendOutput(
            text=text,
            token_ids=token_ids,
            # No sampler ran; the alignment fill keeps the row well-formed and rollout
            # correction is off for demonstrations.
            logprobs=[0.0] * len(token_ids),
            prompt_token_ids=None,
            stop_reason="completed",
            weights_version=None,
        )

    @property
    def remaining(self) -> int:
        return len(self._scripted) - self._next


__all__ = ["ScriptedClient"]
