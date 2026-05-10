from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger("resqai.voice_intelligence")


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None


def _load_whisper():
    try:
        import whisper  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "OpenAI Whisper is not available. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return whisper


@lru_cache(maxsize=4)
def _load_whisper_model(model_name: str, device_str: str):
    whisper = _load_whisper()
    logger.info("loading_whisper model=%s device=%s", model_name, device_str)
    return whisper.load_model(model_name, device=device_str)


def transcribe_audio(
    audio_path: Path,
    *,
    model_name: str,
    device: torch.device,
    language: str | None = None,
) -> TranscriptionResult:
    whisper = _load_whisper()

    # Whisper expects device as string. For CUDA, it can use fp16; for MPS/CPU, fp16 is unsafe.
    device_str = device.type
    fp16 = device.type == "cuda"

    try:
        model = _load_whisper_model(model_name, device_str)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load Whisper model '%s' on device '%s'.", model_name, device_str)
        raise RuntimeError(
            f"Failed to load Whisper model '{model_name}' on device '{device_str}'."
        ) from exc

    try:
        # If `language` is provided, Whisper skips auto language detection.
        # See: whisper.transcribe(..., language="fr")
        kwargs = {"fp16": fp16, "task": "transcribe"}
        if language:
            kwargs["language"] = language
        result = model.transcribe(str(audio_path), **kwargs)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Common failure: missing ffmpeg.
        logger.exception("Whisper transcription failed for '%s'.", str(audio_path))
        raise RuntimeError(
            "Whisper transcription failed. If you see ffmpeg-related errors, install ffmpeg and retry."
        ) from exc

    text = (result.get("text") or "").strip()
    detected_language = result.get("language")
    # If forced language is used, return it for consistency.
    out_lang = language or detected_language
    return TranscriptionResult(text=text, language=out_lang)
