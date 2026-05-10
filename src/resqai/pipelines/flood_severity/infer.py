from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from resqai.pipelines.flood_severity.config import InferConfig
from resqai.pipelines.flood_severity.dataset import IMAGENET_MEAN, IMAGENET_STD
from resqai.pipelines.flood_severity.model import build_efficientnet_b0
from resqai.pipelines.flood_severity.utils import get_device, load_checkpoint, resolve_from_repo

logger = logging.getLogger("resqai.flood_severity")


def build_infer_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@torch.inference_mode()
def predict_single_image(
    image_path: Path,
    checkpoint_path: Path,
    device: torch.device | None = None,
    cfg: InferConfig = InferConfig(),
) -> dict[str, object]:
    if device is None:
        device = get_device()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train first with: python -m resqai.pipelines.flood_severity.train"
        )

    ckpt = load_checkpoint(str(checkpoint_path), device=device)
    class_names: list[str] = ckpt.get("class_names", ["mild", "severe", "no_flood"])
    preprocess = ckpt.get("preprocess", {}) if isinstance(ckpt, dict) else {}
    image_size = int(preprocess.get("image_size", cfg.image_size))

    model = build_efficientnet_b0(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    x = build_infer_transform(image_size)(image).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())

    return {
        "image_path": str(image_path),
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "probabilities": {class_names[i]: float(probs[i].item()) for i in range(len(class_names))},
        "device": device.type,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Infer flood severity from a single image.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/flood_severity/best.pt"))
    p.add_argument("--image-size", type=int, default=InferConfig.image_size)
    args = p.parse_args()

    try:
        result = predict_single_image(
            image_path=resolve_from_repo(args.image),
            checkpoint_path=resolve_from_repo(args.checkpoint),
            cfg=InferConfig(image_size=args.image_size),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("%s", exc)
        raise
    print(result)


if __name__ == "__main__":
    main()
