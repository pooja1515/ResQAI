from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from resqai.pipelines.flood_severity.config import InferConfig
from resqai.pipelines.flood_severity.dataset import IMAGENET_MEAN, IMAGENET_STD
from resqai.pipelines.flood_severity.gradcam import GradCAM
from resqai.pipelines.flood_severity.model import build_efficientnet_b0
from resqai.pipelines.flood_severity.utils import get_device, load_checkpoint, resolve_from_repo
from resqai.pipelines.flood_severity.visualization_utils import (
    ExplanationMetadata,
    save_explanation_artifacts,
)
from torchvision import transforms

logger = logging.getLogger("resqai.flood_severity")


def build_display_transform(image_size: int) -> transforms.Compose:
    # Keep display image spatially aligned with the model input crop.
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
        ]
    )


def build_tensor_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@torch.inference_mode()
def _predict_probs(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return torch.softmax(logits, dim=1).squeeze(0)


def explain_single_image(
    image_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device | None = None,
    cfg: InferConfig = InferConfig(),
) -> dict[str, object]:
    if device is None:
        device = get_device()

    image_path = resolve_from_repo(image_path)
    checkpoint_path = resolve_from_repo(checkpoint_path)
    output_dir = resolve_from_repo(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train first with: python -m resqai.pipelines.flood_severity.train"
        )
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ckpt = load_checkpoint(str(checkpoint_path), device=device)
    class_names: list[str] = ckpt.get("class_names", ["flooded", "non_flooded"])
    preprocess = ckpt.get("preprocess", {}) if isinstance(ckpt, dict) else {}
    image_size = int(preprocess.get("image_size", cfg.image_size))

    model = build_efficientnet_b0(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    raw = Image.open(image_path).convert("RGB")
    display_img = build_display_transform(image_size)(raw)
    x = build_tensor_transform(image_size)(raw).unsqueeze(0).to(device)

    probs = _predict_probs(model, x)
    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())

    # Grad-CAM requires gradients; run it outside inference_mode.
    cam_generator = GradCAM(model=model, target_layer=None)
    try:
        with torch.enable_grad():
            cam_res = cam_generator.generate(x, target_class=pred_idx)
    finally:
        cam_generator.close()

    cam_np = cam_res.cam.numpy().astype(np.float32)

    stem = image_path.stem
    metadata = ExplanationMetadata(
        image_path=str(image_path),
        checkpoint_path=str(checkpoint_path),
        predicted_class=class_names[pred_idx],
        confidence=confidence,
        probabilities={class_names[i]: float(probs[i].item()) for i in range(len(class_names))},
        device=device.type,
        heatmap_path=str(output_dir / f"{stem}_heatmap.png"),
        overlay_path=str(output_dir / f"{stem}_overlay.png"),
    )

    heatmap_path, overlay_path, metadata_path = save_explanation_artifacts(
        output_dir=output_dir,
        stem=stem,
        base_rgb=display_img,
        cam=cam_np,
        metadata=metadata,
    )

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": confidence,
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "metadata_path": str(metadata_path),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Generate Grad-CAM explanation for a single image.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/flood_severity/best.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/explanations"))
    p.add_argument("--image-size", type=int, default=InferConfig.image_size)
    args = p.parse_args()

    try:
        result = explain_single_image(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            cfg=InferConfig(image_size=args.image_size),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("%s", exc)
        raise

    print(result)


if __name__ == "__main__":
    main()
