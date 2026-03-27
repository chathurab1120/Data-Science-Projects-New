"""Preprocessing utilities and dataloaders for chest X-ray classification."""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

CLASS_TO_INDEX: dict[str, int] = {"NORMAL": 0, "PNEUMONIA": 1}
VALID_SPLITS: tuple[str, str, str] = ("train", "val", "test")
VALID_IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
RANDOM_SEED: int = 42


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration from disk.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config content is not a dictionary.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file_pointer:
            loaded_config: Any = yaml.safe_load(file_pointer)
    except yaml.YAMLError:
        raise
    except OSError as exc:
        raise ValueError(f"Unable to read config file: {config_path}") from exc

    if not isinstance(loaded_config, dict):
        raise ValueError("Config content must be a dictionary.")
    return loaded_config


def setup_logging(log_file_path: Path) -> logging.Logger:
    """Configure logger for preprocessing pipeline.

    Args:
        log_file_path: File path where logs should be written.

    Returns:
        Configured logger instance.

    Raises:
        OSError: If log directory cannot be created.
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = logging.getLogger("preprocessing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler: logging.FileHandler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class ChestXRayDataset(Dataset[tuple[Tensor, int, str]]):
    """PyTorch dataset for chest X-ray image classification.

    Args:
        split: Dataset split name, one of train/val/test.
        config: Global project configuration dictionary.
        transform: Optional torchvision transform pipeline.
        logger: Optional logger for warnings and dataset statistics.

    Raises:
        ValueError: If split is invalid or no valid images are found.
        FileNotFoundError: If expected split/class directories are missing.
    """

    def __init__(
        self,
        split: str,
        config: dict[str, Any],
        transform: transforms.Compose | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of: {VALID_SPLITS}.")

        self.split: str = split
        self.transform: transforms.Compose | None = transform
        self.logger: logging.Logger = logger if logger is not None else logging.getLogger("preprocessing")
        self.project_root: Path = Path(__file__).resolve().parents[1]
        self.data_dir: Path = self.project_root / Path(str(config["data"]["data_dir"])) / split
        self.samples: list[tuple[Path, int]] = self._build_samples()

        if not self.samples:
            raise ValueError(f"No valid images found for split '{split}' in {self.data_dir}")

    def _build_samples(self) -> list[tuple[Path, int]]:
        """Build list of valid image samples and labels for the split.

        Args:
            None.

        Returns:
            List of tuples containing image path and class index.

        Raises:
            FileNotFoundError: If class directory does not exist.
        """
        valid_samples: list[tuple[Path, int]] = []
        skipped_count: int = 0

        for class_name, class_index in CLASS_TO_INDEX.items():
            class_dir: Path = self.data_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            image_paths: list[Path] = sorted(
                [
                    file_path
                    for file_path in class_dir.rglob("*")
                    if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_SUFFIXES
                ]
            )

            for image_path in image_paths:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    valid_samples.append((image_path, class_index))
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    skipped_count += 1
                    self.logger.warning("Skipping unreadable image '%s': %s", image_path, exc)

        self.logger.info(
            "Loaded split '%s' with %d valid images and %d skipped images.",
            self.split,
            len(valid_samples),
            skipped_count,
        )
        return valid_samples

    def __len__(self) -> int:
        """Return number of samples in dataset.

        Args:
            None.

        Returns:
            Number of valid image samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        """Retrieve transformed image tensor, label, and image path.

        Args:
            index: Dataset index for the requested sample.

        Returns:
            Tuple of image tensor, integer class label, and image path.

        Raises:
            RuntimeError: If image cannot be opened for reading.
            ValueError: If transformed output is not a tensor.
        """
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as image:
                rgb_image: Image.Image = image.convert("RGB")
                if self.transform is not None:
                    image_tensor: Any = self.transform(rgb_image)
                else:
                    image_tensor = transforms.ToTensor()(rgb_image)
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError("Transform pipeline must return torch.Tensor output.")
        return image_tensor, label, str(image_path)


def get_transforms(config: dict[str, Any]) -> dict[str, transforms.Compose]:
    """Build transform pipelines for train, val, and test splits.

    Args:
        config: Global project configuration dictionary.

    Returns:
        Mapping from split name to torchvision transform pipeline.

    Raises:
        KeyError: If required image normalization keys are missing.
        ValueError: If image size is invalid.
    """
    image_size: int = int(config["image"]["size"])
    if image_size <= 0:
        raise ValueError("Image size must be positive.")

    mean_values: list[float] = [float(value) for value in config["image"]["normalize"]["mean"]]
    std_values: list[float] = [float(value) for value in config["image"]["normalize"]["std"]]

    train_transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values),
        ]
    )
    eval_transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values),
        ]
    )
    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def get_class_weights(train_dataset: ChestXRayDataset) -> Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Args:
        train_dataset: Training dataset instance.

    Returns:
        Tensor with class weights ordered as [NORMAL, PNEUMONIA].

    Raises:
        ValueError: If class counts are invalid for weight computation.
    """
    label_counts: Counter[int] = Counter(label for _, label in train_dataset.samples)
    expected_labels: tuple[int, int] = (0, 1)
    if any(label_counts[label] <= 0 for label in expected_labels):
        raise ValueError("Both classes must have at least one sample to compute class weights.")

    total_samples: int = len(train_dataset.samples)
    class_count: int = len(expected_labels)
    weights: list[float] = [
        total_samples / (class_count * float(label_counts[label_index])) for label_index in expected_labels
    ]
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[DataLoader[tuple[Tensor, int, str]], DataLoader[tuple[Tensor, int, str]], DataLoader[tuple[Tensor, int, str]]]:
    """Create train, val, and test dataloaders using config settings.

    Args:
        config: Global project configuration dictionary.
        logger: Logger for dataset and dataloader metadata.

    Returns:
        Tuple containing train, val, and test dataloaders.

    Raises:
        KeyError: If training batch size key is missing.
        ValueError: If batch size is invalid.
    """
    batch_size: int = int(config["training"]["batch_size"])
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    transform_map: dict[str, transforms.Compose] = get_transforms(config)
    train_dataset: ChestXRayDataset = ChestXRayDataset("train", config, transform_map["train"], logger)
    val_dataset: ChestXRayDataset = ChestXRayDataset("val", config, transform_map["val"], logger)
    test_dataset: ChestXRayDataset = ChestXRayDataset("test", config, transform_map["test"], logger)

    pin_memory: bool = torch.cuda.is_available()
    train_label_counts: Counter[int] = Counter(label for _, label in train_dataset.samples)
    sample_weights: list[float] = [1.0 / float(train_label_counts[label]) for _, label in train_dataset.samples]
    sampler: WeightedRandomSampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader: DataLoader[tuple[Tensor, int, str]] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader: DataLoader[tuple[Tensor, int, str]] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader: DataLoader[tuple[Tensor, int, str]] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    logger.info(
        "Created dataloaders with batch_size=%d, pin_memory=%s, num_workers=0.",
        batch_size,
        pin_memory,
    )
    logger.info(
        "Train label counts used for weighted sampling: NORMAL=%d, PNEUMONIA=%d",
        train_label_counts[0],
        train_label_counts[1],
    )
    return train_loader, val_loader, test_loader


def denormalize_images(image_batch: Tensor, mean_values: list[float], std_values: list[float]) -> Tensor:
    """Denormalize tensor batch for visualization.

    Args:
        image_batch: Normalized image tensor of shape [B, C, H, W].
        mean_values: Channel-wise mean values.
        std_values: Channel-wise standard deviation values.

    Returns:
        Denormalized tensor clipped to [0, 1].
    """
    mean_tensor: Tensor = torch.tensor(mean_values, dtype=image_batch.dtype, device=image_batch.device).view(1, -1, 1, 1)
    std_tensor: Tensor = torch.tensor(std_values, dtype=image_batch.dtype, device=image_batch.device).view(1, -1, 1, 1)
    denormalized: Tensor = image_batch * std_tensor + mean_tensor
    return denormalized.clamp(0.0, 1.0)


def save_augmented_sample_grid(
    image_batch: Tensor,
    labels_batch: Tensor,
    paths_batch: list[str],
    mean_values: list[float],
    std_values: list[float],
    output_path: Path,
) -> None:
    """Save denormalized sample grid to disk.

    Args:
        image_batch: Tensor batch with shape [B, C, H, W].
        labels_batch: Tensor of integer class labels.
        paths_batch: List of source image paths.
        mean_values: Channel-wise normalization mean.
        std_values: Channel-wise normalization std.
        output_path: Output path for figure.

    Returns:
        None.

    Raises:
        OSError: If output file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    denormalized_batch: Tensor = denormalize_images(image_batch, mean_values, std_values)
    sample_count: int = min(8, denormalized_batch.size(0))
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))

    for axis_index, axis in enumerate(axes.flat):
        if axis_index >= sample_count:
            axis.axis("off")
            continue

        image_array = denormalized_batch[axis_index].permute(1, 2, 0).cpu().numpy()
        label_index: int = int(labels_batch[axis_index].item())
        class_name: str = "NORMAL" if label_index == 0 else "PNEUMONIA"
        file_name: str = Path(paths_batch[axis_index]).name
        axis.imshow(image_array)
        axis.set_title(f"{class_name}\n{file_name}", fontsize=9)
        axis.axis("off")

    figure.suptitle("Augmented Training Samples", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def verify_dataloader(
    split_name: str,
    dataloader: DataLoader[tuple[Tensor, int, str]],
    config: dict[str, Any],
    logger: logging.Logger,
    save_grid: bool = False,
) -> None:
    """Validate one dataloader batch and optionally save sample grid.

    Args:
        split_name: Name of data split represented by dataloader.
        dataloader: Dataloader to validate.
        config: Global config dictionary with normalization values.
        logger: Logger for verification details.
        save_grid: Whether to write sample visualization for this split.

    Returns:
        None.

    Raises:
        ValueError: If batch contains NaN or Inf values.
        RuntimeError: If dataloader cannot yield a batch.
    """
    try:
        image_batch, labels_batch, paths_batch = next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError(f"Dataloader for split '{split_name}' is empty.") from exc
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to iterate dataloader for split '{split_name}'.") from exc

    if not torch.isfinite(image_batch).all():
        raise ValueError(f"Detected NaN or Inf values in '{split_name}' batch tensors.")

    label_counter: Counter[int] = Counter(labels_batch.tolist())
    logger.info(
        "Split '%s' batch shape=%s, label_distribution={0:%d, 1:%d}, min=%.4f, max=%.4f",
        split_name,
        tuple(image_batch.shape),
        label_counter.get(0, 0),
        label_counter.get(1, 0),
        float(image_batch.min().item()),
        float(image_batch.max().item()),
    )

    if save_grid:
        mean_values: list[float] = [float(value) for value in config["image"]["normalize"]["mean"]]
        std_values: list[float] = [float(value) for value in config["image"]["normalize"]["std"]]
        output_path: Path = Path(__file__).resolve().parents[1] / "reports" / "figures" / "augmented_samples.png"
        save_augmented_sample_grid(image_batch, labels_batch, list(paths_batch), mean_values, std_values, output_path)
        logger.info("Saved augmented sample visualization to %s", output_path)


def run_preprocessing_checks() -> None:
    """Run full preprocessing pipeline checks and verification.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: If configuration loading or preprocessing checks fail.
    """
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    project_root: Path = Path(__file__).resolve().parents[1]
    logger: logging.Logger = setup_logging(project_root / "reports" / "preprocessing.log")
    config: dict[str, Any] = load_config(project_root / "configs" / "config.yaml")

    train_loader, val_loader, test_loader = get_dataloaders(config, logger)
    verify_dataloader("train", train_loader, config, logger, save_grid=True)
    verify_dataloader("val", val_loader, config, logger)
    verify_dataloader("test", test_loader, config, logger)

    if not isinstance(train_loader.dataset, ChestXRayDataset):
        raise RuntimeError("Training dataloader dataset is not ChestXRayDataset.")
    train_dataset: ChestXRayDataset = train_loader.dataset
    class_weights: Tensor = get_class_weights(train_dataset)
    logger.info("CrossEntropyLoss class weights [NORMAL, PNEUMONIA]: %s", class_weights.tolist())


def main() -> None:
    """Entrypoint for standalone preprocessing checks.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: If any preprocessing step fails.
    """
    try:
        run_preprocessing_checks()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Preprocessing pipeline failed: {exc}") from exc


if __name__ == "__main__":
    main()
