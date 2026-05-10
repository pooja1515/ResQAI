from __future__ import annotations

import argparse
import logging
from pathlib import Path

from resqai.pipelines.voice_intelligence.config import VoiceConfig
from resqai.pipelines.voice_intelligence.distress_analysis import (
    score_distress_heuristic,
    score_distress_transformers,
)
from resqai.pipelines.voice_intelligence.language_utils import normalize_language_code
from resqai.pipelines.voice_intelligence.language_utils import is_supported_language
from resqai.pipelines.voice_intelligence.transcribe import transcribe_audio
from resqai.pipelines.voice_intelligence.utils import dumps_json, ensure_file, get_device

logger = logging.getLogger("resqai.voice_intelligence")


def run_inference(audio_path: Path, cfg: VoiceConfig) -> dict[str, object]:
    ensure_file(audio_path)

    device = get_device()
    forced_lang = normalize_language_code(cfg.language)
    if forced_lang is not None and not is_supported_language(forced_lang):
        raise ValueError("Unsupported --language. Use one of: en, hi, fr, es")

    logger.info(
        "device=%s whisper_model=%s language=%s audio=%s",
        device.type,
        cfg.whisper_model,
        forced_lang or "auto",
        str(audio_path),
    )

    tx = transcribe_audio(
        audio_path,
        model_name=cfg.whisper_model,
        device=device,
        language=forced_lang,
    )
    lang = normalize_language_code(tx.language)

    distress = None
    if cfg.use_transformers:
        distress = score_distress_transformers(tx.text)
    if distress is None:
        distress = score_distress_heuristic(
            tx.text,
            lang,
            base=cfg.distress_base,
            weight=cfg.distress_keyword_weight,
        )

    return {
        "transcription": tx.text,
        "language": lang,
        "distress_score": float(distress.distress_score),
        "urgency": distress.urgency,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    p = argparse.ArgumentParser(description="ResQAI voice intelligence (Whisper + distress scoring).")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--whisper-model", type=str, default=VoiceConfig.whisper_model)
    p.add_argument("--language", type=str, default=None, help="Force Whisper language: en|hi|fr|es")
    p.add_argument("--use-transformers", action="store_true", help="Enable optional transformers distress model.")
    args = p.parse_args()

    cfg = VoiceConfig(
        whisper_model=args.whisper_model,
        language=args.language,
        use_transformers=args.use_transformers,
    )
    out = run_inference(args.audio, cfg)
    print(dumps_json(out))


if __name__ == "__main__":
    main()
