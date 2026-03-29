# All comments are in English.
"""
Random-response adapter for single-turn QA tasks (forward/inverse dynamics).

Reads a JSON file containing a list of candidate response strings and
returns one at random on each call to ``acompletion``.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from PIL import Image

from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.registry import register_adapter


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
        return random.choice(responses)
