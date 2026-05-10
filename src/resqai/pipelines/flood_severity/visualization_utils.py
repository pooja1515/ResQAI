from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ExplanationMetadata:
    image_path: str
    checkpoint_path: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    device: str
    heatmap_path: str
    overlay_path: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """Simple 'jet'-like colormap for x in [0, 1]. Returns uint8 RGB."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def cam_to_heatmap_image(cam: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Convert CAM (H, W) float [0,1] to an RGB heatmap PIL image resized to `size`."""
    cam_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8), mode="L")
    cam_img = cam_img.resize(size, resample=Image.BILINEAR)
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0
    heatmap = _jet_colormap(cam_arr)
    return Image.fromarray(heatmap, mode="RGB")


def overlay_heatmap(base_rgb: Image.Image, heatmap_rgb: Image.Image, alpha: float = 0.45) -> Image.Image:
    if base_rgb.mode != "RGB":
        base_rgb = base_rgb.convert("RGB")
    if heatmap_rgb.mode != "RGB":
        heatmap_rgb = heatmap_rgb.convert("RGB")
    heatmap_rgb = heatmap_rgb.resize(base_rgb.size, resample=Image.BILINEAR)
    return Image.blend(base_rgb, heatmap_rgb, alpha=alpha)


def save_explanation_artifacts(
    output_dir: Path,
    stem: str,
    base_rgb: Image.Image,
    cam: np.ndarray,
    metadata: ExplanationMetadata,
) -> tuple[Path, Path, Path]:
    ensure_dir(output_dir)

    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    metadata_path = output_dir / f"{stem}_metadata.json"

    heatmap_img = cam_to_heatmap_image(cam, size=base_rgb.size)
    overlay_img = overlay_heatmap(base_rgb, heatmap_img, alpha=0.45)

    heatmap_img.save(heatmap_path)
    overlay_img.save(overlay_path)
    save_json(metadata_path, asdict(metadata))

    return heatmap_path, overlay_path, metadata_path

