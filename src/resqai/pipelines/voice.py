from __future__ import annotations


from pathlib import Path

from resqai.pipelines.voice_intelligence.config import VoiceConfig
from resqai.pipelines.voice_intelligence.infer import run_inference


def transcribe_and_analyze_voice(audio_path: str | Path, cfg: VoiceConfig | None = None) -> dict[str, object]:
    """Thin wrapper kept for pipeline compatibility."""
    if cfg is None:
        cfg = VoiceConfig()
    return run_inference(Path(audio_path), cfg)
