from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights

from resqai.pipelines.flood_severity.utils import resolve_from_repo

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLITS = ("train", "val")


def resolve_split_dir(data_dir: Path, split: str) -> Path:
    """Accept either a base dataset dir or an explicit split dir."""
    if data_dir.name == split:
        return data_dir
    return data_dir / split


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_transforms_from_weights(
    weights: EfficientNet_B0_Weights | None,
    image_size: int,
    train: bool,
) -> transforms.Compose:
    # For validation/inference, use the official preprocessing when available.
    if weights is not None and not train:
        return weights.transforms()
    return build_transforms(image_size=image_size, train=train)


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    class_names: list[str]
    train_size: int
    val_size: int


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pretrained: bool = True,
    device_type: str = "cpu",
) -> DataBundle:
    data_dir = resolve_from_repo(data_dir)
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

    train_root = resolve_split_dir(data_dir, "train")
    val_root = resolve_split_dir(data_dir, "val")
    if not train_root.exists():
        raise FileNotFoundError(f"Train directory not found: {train_root}")
    if not val_root.exists():
        raise FileNotFoundError(f"Val directory not found: {val_root}")

    train_ds = datasets.ImageFolder(
        root=str(train_root),
        transform=build_transforms_from_weights(weights, image_size=image_size, train=True),
    )
    val_ds = datasets.ImageFolder(
        root=str(val_root),
        transform=build_transforms_from_weights(weights, image_size=image_size, train=False),
    )

    pin_memory = device_type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    class_names = list(train_ds.classes)
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        train_size=len(train_ds),
        val_size=len(val_ds),
    )
