from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from resqai.pipelines.voice_intelligence.gemma_client import GemmaOllamaClient, OllamaConfig
from resqai.pipelines.voice_intelligence.utils import dumps_json
from resqai.pipelines.multimodal.fusion_utils import load_json_file, parse_fusion_json, validate_inputs
from resqai.pipelines.multimodal.multimodal_prompts import build_fusion_prompt, build_repair_prompt

logger = logging.getLogger("resqai.multimodal_fusion")


@dataclass(frozen=True)
class FusionConfig:
    model: str = "gemma4"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9


def fuse(vision: dict, voice: dict, cfg: FusionConfig, ollama: OllamaConfig | None = None) -> dict[str, object]:
    client = GemmaOllamaClient(cfg=ollama)

    prompt = build_fusion_prompt(vision=vision, voice=voice)
    last_raw: str | None = None
    last_exc: Exception | None = None

    for attempt in range(3):
        raw = client.generate(
            model=cfg.model,
            prompt=prompt if attempt == 0 else build_repair_prompt((last_raw or "")[:4000], vision, voice),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            json_only=True,
        )
        last_raw = raw
        try:
            result = parse_fusion_json(raw)
            return result.to_dict()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Fusion JSON parse/validation failed (attempt %s/3): %s", attempt + 1, exc)

    raise RuntimeError("Gemma did not return valid fusion JSON after retries.") from last_exc


def parse_args() -> tuple[Path, Path, FusionConfig]:
    p = argparse.ArgumentParser(description="ResQAI multimodal fusion agent (vision + voice -> crisis intelligence).")
    p.add_argument("--vision-json", type=Path, required=True)
    p.add_argument("--voice-json", type=Path, required=True)
    p.add_argument("--model", type=str, default=FusionConfig.model, help="Ollama model name (gemma4, gemma3:4b).")
    p.add_argument("--max-new-tokens", type=int, default=FusionConfig.max_new_tokens)
    p.add_argument("--temperature", type=float, default=FusionConfig.temperature)
    p.add_argument("--top-p", type=float, default=FusionConfig.top_p)
    args = p.parse_args()

    cfg = FusionConfig(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return args.vision_json, args.voice_json, cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    vision_path, voice_path, cfg = parse_args()

    if not vision_path.exists():
        raise FileNotFoundError(f"vision-json not found: {vision_path}")
    if not voice_path.exists():
        raise FileNotFoundError(f"voice-json not found: {voice_path}")

    vision = load_json_file(str(vision_path))
    voice = load_json_file(str(voice_path))
    validate_inputs(vision, voice)

    logger.info("ollama_model=%s vision=%s voice=%s", cfg.model, str(vision_path), str(voice_path))
    out = fuse(vision=vision, voice=voice, cfg=cfg)
    print(dumps_json(out))


if __name__ == "__main__":
    main()
