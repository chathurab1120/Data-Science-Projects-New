"""Data exploration pipeline for chest X-ray classification datasets."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

RANDOM_SEED: int = 42
CLASS_NAMES: tuple[str, str] = ("NORMAL", "PNEUMONIA")
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration from disk.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config content is empty or invalid.
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
    """Configure and return project logger.

    Args:
        log_file_path: Destination path for file logging.

    Returns:
        Configured logger instance.

    Raises:
        OSError: If log directory cannot be created.
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = logging.getLogger("data_exploration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler: logging.FileHandler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def list_image_files(directory_path: Path) -> list[Path]:
    """List image files in a directory recursively.

    Args:
        directory_path: Directory containing images.

    Returns:
        Sorted list of image file paths.

    Raises:
        FileNotFoundError: If the directory path does not exist.
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    image_paths: list[Path] = [
        file_path
        for file_path in directory_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(image_paths)


def clean_macos_artifacts(root_dir: Path, logger: logging.Logger) -> int:
    """Delete macOS artifact files recursively.

    Args:
        root_dir: Root directory to clean.
        logger: Active logger for recording file deletions.

    Returns:
        Number of deleted files.

    Raises:
        FileNotFoundError: If root directory does not exist.
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    removed_files_count: int = 0
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        file_name: str = file_path.name
        if file_name == ".DS_Store" or file_name == "._.DS_Store" or file_name.startswith("._"):
            try:
                file_path.unlink()
                removed_files_count += 1
            except OSError as exc:
                logger.warning("Could not remove artifact file %s: %s", file_path, exc)

    logger.info("macOS artifact cleanup completed. Removed %d files.", removed_files_count)
    return removed_files_count


def compute_dataset_distribution(data_dir: Path) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Compute split counts and per-class distributions.

    Args:
        data_dir: Base dataset directory containing train/val/test.

    Returns:
        Tuple of split totals and per-split class distributions.

    Raises:
        FileNotFoundError: If expected split/class directories are missing.
    """
    split_totals: dict[str, int] = {}
    class_distribution: dict[str, dict[str, int]] = {}

    for split_name in SPLIT_NAMES:
        split_path: Path = data_dir / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory not found: {split_path}")

        class_distribution[split_name] = {}
        split_count: int = 0
        for class_name in CLASS_NAMES:
            class_path: Path = split_path / class_name
            if not class_path.exists():
                raise FileNotFoundError(f"Class directory not found: {class_path}")
            class_count: int = len(list_image_files(class_path))
            class_distribution[split_name][class_name] = class_count
            split_count += class_count
        split_totals[split_name] = split_count

    return split_totals, class_distribution


def compute_imbalance_ratio(class_counts: dict[str, int]) -> float:
    """Compute majority/minority imbalance ratio for a class distribution.

    Args:
        class_counts: Mapping of class names to counts.

    Returns:
        Ratio of max class count to min class count.

    Raises:
        ValueError: If class counts are empty or contain zeros.
    """
    if not class_counts:
        raise ValueError("Class counts cannot be empty.")
    values: list[int] = list(class_counts.values())
    if min(values) <= 0:
        raise ValueError("Class counts must be greater than zero.")
    return max(values) / min(values)


def plot_class_distribution(
    class_distribution: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    """Create grouped bar chart for class counts by split.

    Args:
        class_distribution: Class counts grouped by split.
        output_path: Output image file path.

    Returns:
        None.

    Raises:
        OSError: If output cannot be written.
        ValueError: If required splits/classes are missing.
    """
    split_labels: list[str] = list(SPLIT_NAMES)
    normal_counts: list[int] = [class_distribution[split]["NORMAL"] for split in split_labels]
    pneumonia_counts: list[int] = [class_distribution[split]["PNEUMONIA"] for split in split_labels]

    x_values: np.ndarray = np.arange(len(split_labels))
    bar_width: float = 0.36

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x_values - bar_width / 2, normal_counts, bar_width, label="NORMAL")
    axis.bar(x_values + bar_width / 2, pneumonia_counts, bar_width, label="PNEUMONIA")
    axis.set_title("Class Distribution Across Splits")
    axis.set_xlabel("Split")
    axis.set_ylabel("Image Count")
    axis.set_xticks(x_values)
    axis.set_xticklabels(split_labels)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def select_training_samples_for_grid(train_dir: Path, samples_per_class: int = 4) -> list[tuple[str, Path]]:
    """Select random sample images per class for visualization.

    Args:
        train_dir: Path to the training split.
        samples_per_class: Number of images sampled per class.

    Returns:
        List of tuples containing class label and image path.

    Raises:
        ValueError: If any class has fewer images than requested.
        FileNotFoundError: If class directory is missing.
    """
    random.seed(RANDOM_SEED)
    selected_samples: list[tuple[str, Path]] = []

    for class_name in CLASS_NAMES:
        class_dir: Path = train_dir / class_name
        class_files: list[Path] = list_image_files(class_dir)
        if len(class_files) < samples_per_class:
            raise ValueError(
                f"Not enough images in {class_dir}. Required {samples_per_class}, found {len(class_files)}."
            )
        sampled_files: list[Path] = random.sample(class_files, samples_per_class)
        selected_samples.extend((class_name, file_path) for file_path in sampled_files)

    return selected_samples


def plot_sample_images(sample_images: list[tuple[str, Path]], output_path: Path) -> None:
    """Create 2x4 image grid from sampled files.

    Args:
        sample_images: List of (class_name, image_path) tuples.
        output_path: Destination path for output figure.

    Returns:
        None.

    Raises:
        ValueError: If sample size is not exactly 8.
        OSError: If image read/write fails.
        UnidentifiedImageError: If an image file is unreadable.
    """
    if len(sample_images) != 8:
        raise ValueError("sample_images must contain exactly 8 items for a 2x4 grid.")

    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (class_name, image_path) in zip(axes.flat, sample_images):
        with Image.open(image_path) as image:
            axis.imshow(image, cmap="gray" if image.mode in {"L", "LA"} else None)
        axis.set_title(f"{class_name}\n{image_path.name}", fontsize=9)
        axis.axis("off")

    figure.suptitle("Training Samples: 4 NORMAL and 4 PNEUMONIA", fontsize=14)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_image_size_distribution(train_files: list[Path], output_path: Path, sample_size: int = 200) -> None:
    """Create width-vs-height scatter plot for random training images.

    Args:
        train_files: List of all training image paths.
        output_path: Destination path for output figure.
        sample_size: Number of random images to include.

    Returns:
        None.

    Raises:
        ValueError: If there are no training files.
        OSError: If output cannot be written.
    """
    if not train_files:
        raise ValueError("No training files available to plot size distribution.")

    random.seed(RANDOM_SEED)
    selected_files: list[Path] = random.sample(train_files, min(sample_size, len(train_files)))
    widths: list[int] = []
    heights: list[int] = []

    for image_path in selected_files:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                widths.append(width)
                heights.append(height)
        except (UnidentifiedImageError, OSError):
            continue

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(widths, heights, alpha=0.6, s=25)
    axis.set_title("Image Size Distribution (Training Sample)")
    axis.set_xlabel("Width (pixels)")
    axis.set_ylabel("Height (pixels)")
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def compute_image_statistics(train_files: list[Path], sample_size: int = 100) -> dict[str, Any]:
    """Compute channel, dimension, and mode statistics from sampled images.

    Args:
        train_files: List of training image paths.
        sample_size: Number of random images to evaluate.

    Returns:
        Dictionary with pixel stats, dimension range, and mode summary.

    Raises:
        ValueError: If there are no valid images for statistic computation.
    """
    if not train_files:
        raise ValueError("No training files available for statistics.")

    random.seed(RANDOM_SEED)
    selected_files: list[Path] = random.sample(train_files, min(sample_size, len(train_files)))
    channel_means: list[np.ndarray] = []
    channel_stds: list[np.ndarray] = []
    widths: list[int] = []
    heights: list[int] = []
    modes: list[str] = []

    for image_path in selected_files:
        try:
            with Image.open(image_path) as image:
                rgb_image: Image.Image = image.convert("RGB")
                pixel_array: np.ndarray = np.asarray(rgb_image, dtype=np.float32) / 255.0
                flattened_pixels: np.ndarray = pixel_array.reshape(-1, 3)
                channel_means.append(flattened_pixels.mean(axis=0))
                channel_stds.append(flattened_pixels.std(axis=0))
                width, height = image.size
                widths.append(width)
                heights.append(height)
                modes.append(image.mode)
        except (UnidentifiedImageError, OSError):
            continue

    if not channel_means or not channel_stds:
        raise ValueError("No valid images were available for statistics.")

    mean_values: np.ndarray = np.mean(np.vstack(channel_means), axis=0)
    std_values: np.ndarray = np.mean(np.vstack(channel_stds), axis=0)
    mode_counter: Counter[str] = Counter(modes)
    most_common_mode: str = mode_counter.most_common(1)[0][0]

    return {
        "sample_size_used": len(channel_means),
        "pixel_mean": [round(float(value), 6) for value in mean_values.tolist()],
        "pixel_std": [round(float(value), 6) for value in std_values.tolist()],
        "min_dimensions": {"width": min(widths), "height": min(heights)},
        "max_dimensions": {"width": max(widths), "height": max(heights)},
        "most_common_mode": most_common_mode,
    }


def save_summary_json(summary_data: dict[str, Any], output_path: Path) -> None:
    """Save dataset summary dictionary as JSON.

    Args:
        summary_data: Data summary payload.
        output_path: JSON output path.

    Returns:
        None.

    Raises:
        OSError: If file cannot be written.
        TypeError: If summary content is not JSON serializable.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(summary_data, file_pointer, indent=2)


def run_data_exploration() -> None:
    """Execute full data exploration workflow and artifact generation.

    Args:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: If expected data/config paths are missing.
        KeyError: If required config keys are not present.
        ValueError: If dataset content is invalid for required analyses.
    """
    project_root: Path = Path(__file__).resolve().parents[1]
    config_path: Path = project_root / "configs" / "config.yaml"
    reports_dir: Path = project_root / "reports"
    figures_dir: Path = reports_dir / "figures"
    logger: logging.Logger = setup_logging(reports_dir / "data_exploration.log")

    logger.info("Starting data exploration pipeline.")
    config: dict[str, Any] = load_config(config_path)

    data_config: dict[str, Any] = config["data"]
    data_dir: Path = project_root / Path(str(data_config["data_dir"]))
    raw_dir: Path = project_root / "data" / "raw"

    removed_artifacts: int = clean_macos_artifacts(raw_dir, logger)

    split_totals, class_distribution = compute_dataset_distribution(data_dir)
    train_ratio: float = compute_imbalance_ratio(class_distribution["train"])

    logger.info("Split image totals: %s", split_totals)
    logger.info("Class distribution by split: %s", class_distribution)
    logger.info("Training class imbalance ratio (max/min): %.4f", train_ratio)
    logger.warning(
        "Validation set is very small (~16 images) and should not be relied on for validation metrics alone."
    )

    plot_class_distribution(class_distribution, figures_dir / "class_distribution.png")

    train_dir: Path = data_dir / "train"
    sampled_grid_files: list[tuple[str, Path]] = select_training_samples_for_grid(train_dir, samples_per_class=4)
    plot_sample_images(sampled_grid_files, figures_dir / "sample_images.png")

    train_files: list[Path] = list_image_files(train_dir / "NORMAL") + list_image_files(train_dir / "PNEUMONIA")
    plot_image_size_distribution(train_files, figures_dir / "image_size_distribution.png", sample_size=200)

    pixel_statistics: dict[str, Any] = compute_image_statistics(train_files, sample_size=100)
    logger.info("Pixel statistics (100 random train images): %s", pixel_statistics)

    data_summary: dict[str, Any] = {
        "data_dir": str(data_dir),
        "artifact_files_removed_count": removed_artifacts,
        "split_counts": split_totals,
        "class_distribution_per_split": class_distribution,
        "training_imbalance_ratio": round(train_ratio, 6),
        "pixel_statistics": pixel_statistics,
    }

    save_summary_json(data_summary, reports_dir / "data_summary.json")
    logger.info("Saved figures to: %s", figures_dir)
    logger.info("Saved JSON summary to: %s", reports_dir / "data_summary.json")
    logger.info("Data exploration pipeline completed successfully.")


def main() -> None:
    """Program entrypoint for standalone script execution.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: If pipeline fails due to a known processing issue.
    """
    try:
        run_data_exploration()
    except (FileNotFoundError, KeyError, ValueError, yaml.YAMLError, OSError) as exc:
        raise RuntimeError(f"Data exploration failed: {exc}") from exc


if __name__ == "__main__":
    main()
