from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from resqai.pipelines.voice_intelligence.gemma_client import GemmaOllamaClient, OllamaConfig
from resqai.pipelines.voice_intelligence.prompts import build_ollama_prompt, build_repair_prompt
from resqai.pipelines.voice_intelligence.reasoning_utils import SemanticResult, parse_semantic_json
from resqai.pipelines.voice_intelligence.utils import dumps_json

logger = logging.getLogger("resqai.voice_intelligence")


@dataclass(frozen=True)
class ReasonerConfig:
    model: str
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9


class GemmaSemanticReasoner:
    """Gemma semantic reasoner via local Ollama.

    Produces structured JSON for later integration into multimodal reasoning.
    """

    def __init__(self, cfg: ReasonerConfig, ollama: OllamaConfig | None = None) -> None:
        self.cfg = cfg
        self.client = GemmaOllamaClient(cfg=ollama)

    def reason(
        self, transcription: str, language: str | None = None, signals: dict | None = None
    ) -> SemanticResult:
        prompt = build_ollama_prompt(transcription, language, signals=signals)

        # Recovery loop: if the model returns malformed JSON or violates schema,
        # ask it to repair into strict JSON. Keep it bounded.
        last_raw: str | None = None
        last_exc: Exception | None = None
        for attempt in range(3):
            attempt_prompt = prompt
            if attempt == 1:
                # Stricter decoding on retry to reduce schema drift.
                attempt_prompt = build_ollama_prompt(
                    transcription,
                    language,
                    signals={**(signals or {}), "retry": True},
                )
            elif attempt >= 2:
                attempt_prompt = build_repair_prompt((last_raw or "")[:2000])

            raw = self.client.generate(
                model=self.cfg.model,
                prompt=attempt_prompt,
                temperature=self.cfg.temperature if attempt == 0 else 0.0,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_new_tokens,
                json_only=True,
                stop=["\n\n", "```"],
            )
            last_raw = raw
            try:
                return parse_semantic_json(raw)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("Semantic JSON parse/validation failed (attempt %s/3): %s", attempt + 1, exc)
                continue

        raise RuntimeError("Gemma did not return valid structured JSON after retries.") from last_exc


def parse_args() -> tuple[str, str | None, ReasonerConfig]:
    p = argparse.ArgumentParser(description="Semantic emergency reasoning using Gemma (text-only).")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--language", type=str, default=None, help="Optional language code (en|hi|fr|es).")
    p.add_argument(
        "--model",
        type=str,
        default="gemma4",
        help="Ollama model name (gemma4, gemma3:4b).",
    )
    p.add_argument("--max-new-tokens", type=int, default=ReasonerConfig.max_new_tokens)
    p.add_argument("--temperature", type=float, default=ReasonerConfig.temperature)
    p.add_argument("--top-p", type=float, default=ReasonerConfig.top_p)
    args = p.parse_args()

    cfg = ReasonerConfig(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return args.text, args.language, cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    text, language, cfg = parse_args()
    logger.info("ollama_model=%s", cfg.model)

    reasoner = GemmaSemanticReasoner(cfg=cfg)
    result = reasoner.reason(text, language=language)
    print(dumps_json(result.to_dict()))


if __name__ == "__main__":
    main()
