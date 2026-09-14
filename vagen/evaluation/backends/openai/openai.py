from __future__ import annotations
import os
import logging
from typing import Any, Dict, Iterable, List, Tuple
from openai import AsyncAzureOpenAI, AsyncOpenAI
from PIL import Image
from vagen.evaluation.backends._common.base import EvaluationBackend
from vagen.evaluation.backends._common.rendering import apply_cache_breakpoints, pil_to_dataurl_png, compile_text_images_for_order
from vagen.evaluation.backends._common.registry import register_adapter, register_client


@register_client("openai", "openai_responses")
def build_client_openai(cfg: Dict[str, Any]) -> AsyncOpenAI:
    api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
    base_url = cfg.get("base_url")
    return AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)


@register_client("azure", "azure_responses")
def build_client_azure(cfg: Dict[str, Any]) -> AsyncAzureOpenAI:
    endpoint = cfg.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = cfg.get("azure_api_key") or os.getenv("AZURE_OPENAI_API_KEY", "") or os.getenv("AZURE_API_KEY", "")
    api_version = cfg.get("azure_api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not endpoint or not api_key:
        raise ValueError("Azure endpoint/api_key missing.")
    return AsyncAzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)

logger = logging.getLogger(__name__)


def _log_usage(resp) -> None:
    """One INFO line per call with what the endpoint billed, so a run's cost and its
    prompt-cache hit rate can be read off the job log (OpenRouter fills ``cost`` when the
    request asks for ``usage: {include: true}``)."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    pd = getattr(usage, "prompt_tokens_details", None)
    cd = getattr(usage, "completion_tokens_details", None)
    extra = getattr(usage, "model_extra", None) or {}
    logger.info(
        "usage prompt=%s cached=%s completion=%s reasoning=%s cost=%s",
        getattr(usage, "prompt_tokens", None),
        getattr(pd, "cached_tokens", None) if pd else None,
        getattr(usage, "completion_tokens", None),
        getattr(cd, "reasoning_tokens", None) if cd else None,
        extra.get("cost"),
    )


@register_adapter("openai", "azure")
class OpenAIAdapter(EvaluationBackend):
    """
    OpenAI-compatible multimodal adapter:
    - messages use content parts with {"type": "text"} and {"type": "image_url"}.
    - capability flags allow omitting unsupported kwargs (e.g., o3).
    """

    def __init__(
        self,
        client,
        model: str,
        cache_control: bool = False,
    ):
        self.client = client
        self.model = model
        #: Attach ``cache_control`` to the text before a harness's cache breakpoint. Off by
        #: default: OpenAI proper and most self-hosted servers reject unknown part fields;
        #: OpenRouter (Gemini, Anthropic, ...) uses it for explicit prompt caching.
        self.cache_control = bool(cache_control)


    def _segments_to_content(self, segs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for kind, val in segs:
            if kind == "text":
                if str(val).strip():
                    content.append({"type": "text", "text": str(val)})
            else:
                content.append({"type": "image_url", "image_url": {"url": pil_to_dataurl_png(val)}})
        return apply_cache_breakpoints(content, self.cache_control)

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "system", "content": self._segments_to_content(segs)}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "user", "content": self._segments_to_content(segs)}

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **chat_config,
        )
        _log_usage(resp)
        return resp.choices[0].message.content or ""
