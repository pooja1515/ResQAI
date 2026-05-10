from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PrepConfig:
    src_root: Path
    dst_root: Path
    seed: int = 42
    val_ratio: float = 0.2
    dry_run: bool = False

    @property
    def labeled_root(self) -> Path:
        # Actual dataset format:
        # train/Labeled/Flooded/image
        # train/Labeled/Non-Flooded/image
        return self.src_root / "train" / "Labeled"


def first_existing_dir(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def split_items(items: list[Path], *, seed: int, val_ratio: float) -> tuple[list[Path], list[Path]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_val = int(round(len(items) * val_ratio))
    val_items = items[:n_val]
    train_items = items[n_val:]
    return train_items, val_items


def safe_copy(src: Path, dst_dir: Path, *, dry_run: bool) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        if not dry_run:
            shutil.copy2(src, dst)
        return dst

    # Name collision: append incremental suffix.
    stem = src.stem
    suffix = src.suffix
    i = 1
    while True:
        candidate = dst_dir / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            if not dry_run:
                shutil.copy2(src, candidate)
            return candidate
        i += 1


def prepare(cfg: PrepConfig) -> None:
    if not cfg.labeled_root.exists():
        raise FileNotFoundError(
            f"Expected labeled data at: {cfg.labeled_root}. "
            "Update --src-root to point at the dataset root containing train/Labeled/."
        )

    flooded_image_dir = first_existing_dir(
        cfg.labeled_root / "Flooded" / "image",
        cfg.labeled_root / "flooded" / "image",
    )
    non_flooded_image_dir = first_existing_dir(
        cfg.labeled_root / "Non-Flooded" / "image",
        cfg.labeled_root / "Non-Flooded" / "images",
        cfg.labeled_root / "non-flooded" / "image",
        cfg.labeled_root / "non_flooded" / "image",
    )

    if flooded_image_dir is None:
        raise FileNotFoundError(
            f"Flooded image directory not found under: {cfg.labeled_root} "
            "(expected something like Labeled/Flooded/image)"
        )
    if non_flooded_image_dir is None:
        raise FileNotFoundError(
            f"Non-Flooded image directory not found under: {cfg.labeled_root} "
            "(expected something like Labeled/Non-Flooded/image)"
        )

    # IMPORTANT:
    # - Use ONLY "image/" folders
    # - Ignore "mask/" folders entirely
    # - Ignore unlabeled data and any original val/test splits by not referencing them.
    class_dirs = {
        "flooded": flooded_image_dir,
        "non_flooded": non_flooded_image_dir,
    }

    all_by_class: dict[str, list[Path]] = {k: iter_images(v) for k, v in class_dirs.items()}
    if any(len(v) == 0 for v in all_by_class.values()):
        missing = [k for k, v in all_by_class.items() if len(v) == 0]
        raise FileNotFoundError(
            "No images found for class(es): "
            + ", ".join(missing)
            + f". Looked under: {cfg.labeled_root}"
        )

    # Split per class to keep label balance.
    split: dict[str, dict[str, list[Path]]] = {"train": {}, "val": {}}
    for class_name, items in all_by_class.items():
        train_items, val_items = split_items(items, seed=cfg.seed, val_ratio=cfg.val_ratio)
        split["train"][class_name] = train_items
        split["val"][class_name] = val_items

    # Copy files into ImageFolder-friendly structure:
    # datasets/flood_images/{train,val}/{flooded,non_flooded}/...
    copies: dict[str, dict[str, int]] = {"train": {"flooded": 0, "non_flooded": 0}, "val": {"flooded": 0, "non_flooded": 0}}
    for split_name in ("train", "val"):
        for class_name in ("flooded", "non_flooded"):
            out_dir = cfg.dst_root / split_name / class_name
            for src_path in split[split_name][class_name]:
                safe_copy(src_path, out_dir, dry_run=cfg.dry_run)
                copies[split_name][class_name] += 1

    # Stats
    total_train = sum(copies["train"].values())
    total_val = sum(copies["val"].values())
    print("=== Flood dataset preparation ===")
    print(f"src_root: {cfg.src_root}")
    print(f"labeled_root: {cfg.labeled_root}")
    print(f"flooded_images: {class_dirs['flooded']}")
    print(f"non_flooded_images: {class_dirs['non_flooded']}")
    print(f"dst_root: {cfg.dst_root}")
    print(f"seed: {cfg.seed}  val_ratio: {cfg.val_ratio}")
    print(f"dry_run: {cfg.dry_run}")
    print("")
    print("Split counts:")
    print(f"  train: {total_train}  (flooded={copies['train']['flooded']}, non_flooded={copies['train']['non_flooded']})")
    print(f"  val:   {total_val}  (flooded={copies['val']['flooded']}, non_flooded={copies['val']['non_flooded']})")
    print("")
    if not cfg.dry_run:
        print("Output structure created for torchvision.datasets.ImageFolder:")
        print(f"  {cfg.dst_root}/train/flooded/")
        print(f"  {cfg.dst_root}/train/non_flooded/")
        print(f"  {cfg.dst_root}/val/flooded/")
        print(f"  {cfg.dst_root}/val/non_flooded/")


def parse_args() -> PrepConfig:
    p = argparse.ArgumentParser(
        description="Prepare an ImageFolder-compatible flood dataset from labeled data only."
    )
    p.add_argument(
        "--src-root",
        type=Path,
        default=Path("dataset") / "image" / "flood_images",
        help="Path to the current dataset root containing train/Labeled/...",
    )
    p.add_argument(
        "--dst-root",
        type=Path,
        default=Path("dataset") / "image" / "flood_images_processed",
        help="Destination dataset root (ImageFolder format).",
    )
    p.add_argument("--seed", type=int, default=42, help="Reproducible shuffle seed.")
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction (per class).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print stats without copying files.")
    args = p.parse_args()

    return PrepConfig(
        src_root=args.src_root,
        dst_root=args.dst_root,
        seed=args.seed,
        val_ratio=args.val_ratio,
        dry_run=args.dry_run,
    )


def main() -> None:
    cfg = parse_args()
    prepare(cfg)


if __name__ == "__main__":
    main()
