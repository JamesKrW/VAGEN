"""Prompt-cache plumbing of the evaluation client: breakpoint parts, cache_control, session ids."""
import pytest
from PIL import Image

from vagen.evaluation.backends._common.rendering import CACHE_BREAKPOINT, apply_cache_breakpoints
from vagen.evaluation.backends.openai.openai import OpenAIAdapter
from vagen.evaluation.client import ChatClient, _text_and_images


def test_breakpoint_part_renders_to_the_marker_and_images_stay_aligned():
    msg = {"role": "user", "content": [{"type": "text", "text": "a "}, {"type": "image"},
                                       {"type": "cache_breakpoint"}, {"type": "text", "text": " b"}],
           "images": ["IMG"]}
    text, images = _text_and_images(msg)
    assert text == f"a <image>{CACHE_BREAKPOINT} b" and images == ["IMG"]


def test_enabled_attaches_cache_control_to_the_text_before_the_marker():
    content = [{"type": "text", "text": f"stable{CACHE_BREAKPOINT}fresh"}, {"type": "image_url", "image_url": {"url": "x"}}]
    out = apply_cache_breakpoints(content, enabled=True)
    assert out[0] == {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}
    assert out[1] == {"type": "text", "text": "fresh"} and out[2]["type"] == "image_url"


def test_disabled_strips_the_marker():
    out = apply_cache_breakpoints([{"type": "text", "text": f"stable{CACHE_BREAKPOINT}fresh"}], enabled=False)
    assert out == [{"type": "text", "text": "stable"}, {"type": "text", "text": "fresh"}]
    assert apply_cache_breakpoints([{"type": "text", "text": "plain"}], enabled=True) == [{"type": "text", "text": "plain"}]


def test_openai_adapter_honours_the_flag():
    text = f"TARGET <image>{CACHE_BREAKPOINT}turn 1 view <image>"
    imgs = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]
    on = OpenAIAdapter(client=None, model="m", cache_control=True).format_user_turn(text, imgs)
    assert any("cache_control" in p for p in on["content"])
    off = OpenAIAdapter(client=None, model="m").format_user_turn(text, imgs)
    assert not any("cache_control" in p for p in off["content"])
    assert not any(CACHE_BREAKPOINT in p.get("text", "") for p in off["content"])


@pytest.mark.asyncio
async def test_session_id_auto_becomes_one_id_per_client():
    seen = []

    class Adapter:
        def format_system(self, text, images): return {"role": "system", "content": text}
        def format_user_turn(self, text, images): return {"role": "user", "content": text}
        def format_assistant_turn(self, text): return {"role": "assistant", "content": text}
        async def acompletion(self, messages, **cfg):
            seen.append(cfg); return "ok"

    a = ChatClient(Adapter(), chat_config={"extra_body": {"session_id": "auto", "reasoning": {"effort": "low"}}})
    b = ChatClient(Adapter(), chat_config={"extra_body": {"session_id": "auto"}})
    for c in (a, a, b):
        cid = c._open(None)
        c.encode([{"role": "user", "content": "hi"}])
        await c.generate([0])
        c._api_messages.pop(cid, None)
    ids = [cfg["extra_body"]["session_id"] for cfg in seen]
    assert ids[0] == ids[1] != "auto" and ids[2] not in (ids[0], "auto")
    assert seen[0]["extra_body"]["reasoning"] == {"effort": "low"}   # the rest of extra_body survives


def test_image_url_extra_is_merged_into_every_image_part():
    imgs = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]
    msg = OpenAIAdapter(client=None, model="m", image_url_extra={"max_dynamic_patch": 1}).format_user_turn(
        "a <image> b <image>", imgs)
    parts = [p for p in msg["content"] if p["type"] == "image_url"]
    assert len(parts) == 2 and all(p["image_url"]["max_dynamic_patch"] == 1 and p["image_url"]["url"].startswith("data:") for p in parts)
    plain = OpenAIAdapter(client=None, model="m").format_user_turn("a <image>", imgs[:1])
    assert set(plain["content"][-1]["image_url"]) == {"url"}
