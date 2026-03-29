# All comments are in English.
"""
Random-navigation adapter for multi-turn active-exploration tasks.

Strategy:
  - Turn 1: ``select_view(init_view)`` then a few random movement actions.
  - Middle turns: random movement/rotation actions.
  - Last turn: parse all camera poses seen so far from the conversation
    history and submit a randomly chosen one as the answer.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.registry import register_adapter

# Movement and rotation primitives available in the ScannetTool environment.
MOVE_ACTIONS = [
    "move_forward", "move_backward",
    "move_right", "move_left",
    "move_up", "move_down",
]
ROTATE_ACTIONS = [
    "turn_left", "turn_right",
    "look_up", "look_down",
    "rotate_ccw", "rotate_cw",
]
ALL_ACTIONS = MOVE_ACTIONS + ROTATE_ACTIONS

# Regex to extract "Step X/Y" from observation text.
_STEP_RE = re.compile(r"Step\s+(\d+)\s*/\s*(\d+)")
# Regex to extract camera 6-DoF pose from observation text.
_POSE_RE = re.compile(
    r"\[tx=([-\d.]+),\s*ty=([-\d.]+),\s*tz=([-\d.]+),\s*"
    r"rx=([-\d.]+)°?,\s*ry=([-\d.]+)°?,\s*rz=([-\d.]+)°?\]"
)


def _extract_text(message: Dict[str, Any]) -> str:
    """Pull plain text out of an adapter-formatted message."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _parse_step_info(text: str) -> Optional[Tuple[int, int]]:
    """Return (current_step, max_steps) or None."""
    m = _STEP_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _parse_poses_from_messages(messages: List[Dict[str, Any]]) -> List[Tuple[float, ...]]:
    """Extract all 6-DoF camera poses reported in user messages."""
    poses: List[Tuple[float, ...]] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = _extract_text(msg)
        for m in _POSE_RE.finditer(text):
            pose = tuple(float(m.group(i)) for i in range(1, 7))
            poses.append(pose)
    return poses


def _format_pose(pose: Tuple[float, ...]) -> str:
    """Format a 6-DoF pose tuple into the answer() argument string."""
    return ", ".join(f"{v:.2f}" for v in pose)


def _random_move_actions(n: int = 3) -> List[str]:
    """Generate a short sequence of random movement/rotation actions."""
    count = random.randint(1, n)
    return [random.choice(ALL_ACTIONS) for _ in range(count)]


@register_adapter("random_navigation")
class RandomNavigationAdapter(ModelAdapter):
    """Random-walk agent that submits a random observed pose on the last turn."""

    def __init__(self, *, client: Any = None, model: Any = None, **kwargs: Any) -> None:
        pass

    # ── formatting ──

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        return {"role": "system", "content": [{"type": "text", "text": text}]}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    # ── completion ──

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        # Determine current step and max steps from the latest user message.
        last_user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_text = _extract_text(msg)
                break

        step_info = _parse_step_info(last_user_text)
        current_step = step_info[0] if step_info else 1
        max_steps = step_info[1] if step_info else 1

        is_last_turn = current_step >= max_steps

        if is_last_turn:
            # Collect all camera poses observed during the episode.
            poses = _parse_poses_from_messages(messages)
            if poses:
                chosen = random.choice(poses)
            else:
                # Fallback: submit a zero pose.
                chosen = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            actions_str = f"answer({_format_pose(chosen)})"
        elif current_step == 1:
            # First turn: must select a view before moving.
            moves = _random_move_actions(3)
            actions_str = "|".join(["select_view(init_view)"] + moves)
        else:
            # Middle turns: random movements.
            moves = _random_move_actions(3)
            actions_str = "|".join(moves)

        return f"<think>random exploration step {current_step}/{max_steps}</think><action>{actions_str}|</action>"
