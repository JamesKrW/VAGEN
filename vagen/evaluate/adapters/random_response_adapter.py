# All comments are in English.
"""
Random-response adapter for single-turn QA tasks (forward/inverse dynamics).

Reads a JSON file containing a list of candidate response strings and
returns one at random on each call to ``acompletion``.

Reproducibility: the RNG is seeded with (env_seed, message_hash),
so identical inputs always produce the same output.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Optional

from PIL import Image

from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.registry import register_adapter


def _messages_fingerprint(messages: List[Dict[str, Any]]) -> int:
    """Derive a deterministic seed from conversation content (text + images)."""
    h = hashlib.sha256()
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            h.update(content.encode())
        elif isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    h.update(p.get("text", "").encode())
                elif p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    h.update(url.encode())
    return int(h.hexdigest()[:16], 16)


@register_adapter("random_response")
class RandomResponseAdapter(ModelAdapter):
    """Pick a random response from a pre-defined list loaded from a JSON file."""

    def __init__(self, *, client: Any = None, model: Any = None, **kwargs: Any) -> None:
        # client/model are unused but accepted for registry compatibility
        pass

    # ── formatting (never actually used, but required by ABC) ──

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        return {"role": "system", "content": [{"type": "text", "text": text}]}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    # ── completion ──

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        file_path: Optional[str] = chat_config.get("file_path")
        if not file_path:
            raise ValueError(
                "RandomResponseAdapter requires 'file_path' in chat_config "
                "pointing to a JSON list of response strings."
            )
        with open(file_path, "r", encoding="utf-8") as f:
            responses: List[str] = json.load(f)
        if not responses:
            raise ValueError(f"Response list in {file_path} is empty.")
        base_seed = int(chat_config.get("random_seed", 0))
        rng = random.Random(base_seed ^ _messages_fingerprint(messages))
        return rng.choice(responses)
