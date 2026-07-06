from __future__ import annotations

import base64
from pathlib import Path

import requests

from .vision import (
    DEFAULT_REQUEST_TIMEOUT,
    build_icon_prompt,
    is_local_url,
    parse_semantic_response,
    request_with_retries,
)


class OllamaVisionClient:
    """Local-only vision client for an Ollama instance (native ``/api/generate``)."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not is_local_url(self.base_url):
            raise ValueError("Ollama endpoint must be local only: localhost, 127.0.0.1, or ::1")

    def caption_icon(self, model: str, image_path: Path) -> str:
        semantic = self.identify_icon(model, image_path)
        return str(semantic.get("label", "")).strip().lower()

    def identify_icon(self, model: str, image_path: Path) -> dict[str, object]:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = build_icon_prompt(model)
        response = request_with_retries(
            lambda: requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=self.timeout,
            )
        )
        response.raise_for_status()
        data = response.json()
        text = (data.get("response") or "").strip().lower()
        return parse_semantic_response(text)

    @staticmethod
    def is_local_url(base_url: str) -> bool:
        return is_local_url(base_url)
