from __future__ import annotations

import base64
from pathlib import Path

import requests

from .vision import (
    DEFAULT_REQUEST_TIMEOUT,
    build_icon_prompt,
    image_mime_type,
    is_local_url,
    parse_semantic_response,
    request_with_retries,
)


class LlamaCppVisionClient:
    """Local-only vision client for a llama.cpp ``llama-server`` instance.

    Talks to the OpenAI-compatible ``/v1/chat/completions`` endpoint using
    base64 ``image_url`` content parts, which is the stable multimodal
    interface exposed by llama-server.
    """

    def __init__(self, base_url: str, timeout: float = DEFAULT_REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not is_local_url(self.base_url):
            raise ValueError("llama.cpp endpoint must be local only: localhost, 127.0.0.1, or ::1")

    def caption_icon(self, model: str, image_path: Path) -> str:
        semantic = self.identify_icon(model, image_path)
        return str(semantic.get("label", "")).strip().lower()

    def identify_icon(self, model: str, image_path: Path) -> dict[str, object]:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_uri = f"data:{image_mime_type(image_path)};base64,{image_b64}"
        prompt = build_icon_prompt(model)
        payload: dict[str, object] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "stream": False,
            "temperature": 0,
        }
        # llama-server serves a single model and ignores this field, but the
        # OpenAI schema expects it and remote proxies may route on it.
        if model:
            payload["model"] = model
        response = request_with_retries(
            lambda: requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
        )
        response.raise_for_status()
        data = response.json()
        text = self._extract_message_text(data)
        return parse_semantic_response(text)

    @staticmethod
    def _extract_message_text(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip().lower()
        # Some servers return content as a list of parts.
        if isinstance(content, list):
            parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
            return " ".join(parts).strip().lower()
        return ""

    @staticmethod
    def is_local_url(base_url: str) -> bool:
        return is_local_url(base_url)
