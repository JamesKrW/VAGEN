from __future__ import annotations
from typing import Any, Dict, List, Tuple
from PIL import Image
from vagen.evaluation.backends._common.base import EvaluationBackend
from vagen.evaluation.backends._common.rendering import apply_cache_breakpoints, pil_to_dataurl_png, compile_text_images_for_order
from vagen.evaluation.backends._common.registry import register_adapter


@register_adapter("openai_responses", "azure_responses")
class OpenAIResponsesAdapter(EvaluationBackend):
    """
    Adapter for OpenAI Responses API (client.responses.create).
    Required for models like gpt-5.4-pro that only support the Responses API.
    """

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def _segments_to_content(self, segs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for kind, val in segs:
            if kind == "text":
                if str(val).strip():
                    content.append({"type": "input_text", "text": str(val)})
            else:
                content.append({"type": "input_image", "image_url": pil_to_dataurl_png(val)})
        return apply_cache_breakpoints(content, enabled=False, text_type="input_text")

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "system", "content": self._segments_to_content(segs)}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "user", "content": self._segments_to_content(segs)}

    def format_assistant_turn(self, text: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}

    #: Chat-completions parameter -> its Responses name. The rest of a chat_config written
    #: for /chat/completions (an OpenRouter `extra_body`, `stream`, ...) is not a Responses
    #: parameter and is dropped rather than raising TypeError inside the SDK.
    _RENAMES = {"max_tokens": "max_output_tokens", "max_completion_tokens": "max_output_tokens",
                "max_new_tokens": "max_output_tokens"}
    _PASS_THROUGH = ("max_output_tokens", "reasoning", "text", "metadata", "store",
                     "truncation", "parallel_tool_calls", "tools", "tool_choice", "timeout")

    def _responses_kwargs(self, chat_config: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a chat_config into Responses parameters.

        ``temperature`` and ``top_p`` are dropped: the reasoning models this API is required
        for (gpt-5.x-pro, gpt-6-astra) reject them, and a 0.0 temperature written for the
        chat endpoint should not turn every call into a 400. ``extra_body.reasoning`` is the
        OpenRouter spelling of the effort knob and becomes the native ``reasoning`` argument.
        """
        out: Dict[str, Any] = {}
        for key, value in chat_config.items():
            if value is None:
                continue
            if key in self._RENAMES:
                out[self._RENAMES[key]] = value
            elif key in self._PASS_THROUGH:
                out[key] = value
            elif key == "extra_body" and isinstance(value, dict):
                reasoning = value.get("reasoning")
                if isinstance(reasoning, dict) and reasoning.get("effort"):
                    out.setdefault("reasoning", {"effort": reasoning["effort"]})
        # Ask for the reasoning summary as well: the raw thinking never leaves the API, the
        # summary is the only trace of it we can store (raw_responses.json).
        if not isinstance(out.get("reasoning"), dict):
            out["reasoning"] = {}
        out["reasoning"].setdefault("summary", "auto")
        return out

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        kwargs = self._responses_kwargs(chat_config)
        try:
            resp = await self.client.responses.create(model=self.model, input=messages, **kwargs)
        except Exception as e:  # noqa: BLE001 - a deployment that rejects `summary` gets one retry without it
            if "summary" in str(e).lower() and isinstance(kwargs.get("reasoning"), dict) and "summary" in kwargs["reasoning"]:
                kwargs["reasoning"] = {k: v for k, v in kwargs["reasoning"].items() if k != "summary"}
                resp = await self.client.responses.create(model=self.model, input=messages, **kwargs)
            else:
                raise
        try:
            self.last_raw = resp.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            self.last_raw = {"repr": repr(resp)}
        return resp.output_text or ""
