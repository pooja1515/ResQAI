from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logger = logging.getLogger("resqai.voice_intelligence")


SUPPORTED_OLLAMA_MODELS = {"gemma4", "gemma3:4b"}


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 120.0
    retries: int = 2
    backoff_s: float = 1.0


class OllamaError(RuntimeError):
    pass


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


class GemmaOllamaClient:
    def __init__(self, cfg: OllamaConfig | None = None) -> None:
        self.cfg = cfg or OllamaConfig()

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        json_only: bool = True,
        stop: list[str] | None = None,
    ) -> str:
        if model not in SUPPORTED_OLLAMA_MODELS:
            raise ValueError(f"Unsupported model {model!r}. Use one of: {sorted(SUPPORTED_OLLAMA_MODELS)}")

        url = f"{self.cfg.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens,
            },
        }
        if stop:
            # Ollama supports stop sequences via options.stop (list of strings).
            payload["options"]["stop"] = stop
        if json_only:
            # Ollama enforces JSON output formatting for supported models.
            payload["format"] = "json"

        last_exc: Exception | None = None
        for attempt in range(self.cfg.retries + 1):
            try:
                out = _post_json(url, payload, timeout_s=self.cfg.timeout_s)
                resp = out.get("response")
                if not isinstance(resp, str) or not resp.strip():
                    raise OllamaError(f"Empty response from Ollama at {url}")
                return resp.strip()
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OllamaError) as exc:
                last_exc = exc
                if attempt >= self.cfg.retries:
                    break
                sleep_s = self.cfg.backoff_s * (2**attempt)
                logger.warning("Ollama request failed (attempt %s/%s). Retrying in %.1fs: %s", attempt + 1, self.cfg.retries + 1, sleep_s, exc)
                time.sleep(sleep_s)

        raise OllamaError(
            "Failed to get a response from Ollama. "
            "Make sure Ollama is running and the model is pulled (e.g., `ollama pull gemma4`)."
        ) from last_exc


def is_ollama_running(base_url: str = "http://127.0.0.1:11434", timeout_s: float = 3.0) -> bool:
    try:
        req = Request(f"{base_url}/api/tags", method="GET")
        with urlopen(req, timeout=timeout_s):  # noqa: S310
            return True
    except Exception:
        return False
