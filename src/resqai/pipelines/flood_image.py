from __future__ import annotations

from pathlib import Path

from resqai.pipelines.flood_severity.infer import predict_single_image


def analyze_flood_image(image_path: str | Path, checkpoint_path: str | Path) -> dict[str, object]:
    """Thin wrapper kept for pipeline compatibility.

    This delegates to the dedicated `flood_severity` package.
    """

    return predict_single_image(Path(image_path), Path(checkpoint_path))
