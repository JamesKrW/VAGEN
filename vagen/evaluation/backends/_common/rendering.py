"""Shared multimodal rendering helpers for evaluation backends."""

from __future__ import annotations
import base64
import io
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image

IMAGE_PLACEHOLDER = "<image>"
#: A harness puts this between two text parts to say "everything before here is a stable
#: prefix worth caching". Adapters that speak Anthropic-style ``cache_control`` (OpenAI-
#: compatible routers such as OpenRouter, Anthropic) attach it to the text before the
#: marker; every other adapter strips it. In the message list it is the content part
#: ``{"type": "cache_breakpoint"}``; ``ChatClient`` renders that to this string.
CACHE_BREAKPOINT = "<cache_breakpoint>"
CACHE_BREAKPOINT_TYPE = "cache_breakpoint"


def apply_cache_breakpoints(content: List[Dict[str, Any]], enabled: bool,
                            text_type: str = "text") -> List[Dict[str, Any]]:
    """Resolve ``CACHE_BREAKPOINT`` markers inside OpenAI-style text parts.

    ``enabled``: the text before a marker becomes its own part carrying
    ``cache_control: {"type": "ephemeral"}``; disabled: the marker is just removed.
    Only the LAST breakpoint matters to Gemini via OpenRouter, Anthropic honours up to four;
    emitting all of them is harmless.
    """
    out: List[Dict[str, Any]] = []
    for part in content:
        text = part.get("text") if part.get("type") == text_type else None
        if not text or CACHE_BREAKPOINT not in text:
            out.append(part)
            continue
        chunks = text.split(CACHE_BREAKPOINT)
        for i, chunk in enumerate(chunks):
            is_prefix = i < len(chunks) - 1
            if not chunk:
                # A marker right after an image (or another marker): the cached prefix ends
                # with the previous part, so that part carries the control.
                if is_prefix and enabled and out:
                    out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
                continue
            new_part = {**part, "text": chunk}
            if is_prefix and enabled:
                new_part["cache_control"] = {"type": "ephemeral"}
            out.append(new_part)
    return out

def pil_to_dataurl_png(img: Image.Image) -> str:
    """Encode a PIL image as a data URL (PNG)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def parse_data_url(data_url: str) -> Optional[Tuple[str, str]]:
    """
    Parse data URL. Return (mime_type, base64_string) or None if not data URL.
    Example: 'data:image/png;base64,AAAA...'
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None
    try:
        header, b64 = data_url.split(",", 1)
        # header like: data:image/png;base64
        mime = header.split(";")[0].split(":")[1]
        return mime, b64
    except Exception:
        return None

def compile_text_images_for_order(text: str, images: List[Image.Image], placeholder: str = IMAGE_PLACEHOLDER) -> List[Tuple[str, Any]]:
    """
    Split text by placeholder and interleave with images preserving order.
    Returns a list of segments: [("text", str) | ("image", PIL.Image)]
    """
    parts = text.split(placeholder)
    segs: List[Tuple[str, Any]] = []
    for i, p in enumerate(parts):
        if p:
            segs.append(("text", p))
        if i < len(parts) - 1:
            if i < len(images):
                segs.append(("image", images[i]))
            else:
                segs.append(("text", ""))
    if len(images) > max(0, len(parts) - 1):
        for img in images[len(parts) - 1:]:
            segs.append(("image", img))
    return segs

def extract_images(obs: Dict[str, Any]) -> List[Image.Image]:
    """
    Heuristically extract images from an observation dict.
    - Prefer obs["multi_modal_input"]["<image>"] if present.
    - Else obs["images"] if it's a list of PIL.Image.
    - Else return [].
    """
    try:
        mm = obs.get("multi_modal_input", {})
        imgs = mm.get(IMAGE_PLACEHOLDER, [])
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], Image.Image):
            return imgs
    except Exception:
        pass
    imgs = obs.get("images")
    if isinstance(imgs, list) and (not imgs or isinstance(imgs[0], Image.Image)):
        return imgs
    return []

def _now_tag() -> str:
    """Return a compact timestamp tag."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")
