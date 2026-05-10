from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path("datasets/flood_images")
    output_dir: Path = Path("artifacts/flood_severity")

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2

    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    seed: int = 42
    pretrained: bool = True

    # When True, uses AMP on CUDA only.
    mixed_precision: bool = True

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.data_dir / "val"

    @property
    def best_ckpt_path(self) -> Path:
        return self.output_dir / "best.pt"


@dataclass(frozen=True)
class InferConfig:
    image_size: int = 224
