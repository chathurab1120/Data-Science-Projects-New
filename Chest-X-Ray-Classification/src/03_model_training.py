"""Model training pipeline for chest X-ray pneumonia classification."""

from __future__ import annotations

import importlib.util
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, densenet121
from tqdm import tqdm

RANDOM_SEED: int = 42


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration from disk.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config content is invalid.
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
    """Set up console and file logging.

    Args:
        log_file_path: Path for the training log file.

    Returns:
        Configured logger instance.

    Raises:
        OSError: If log directory cannot be created.
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = logging.getLogger("model_training")
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


class ChestXRayModel(nn.Module):
    """DenseNet121 transfer-learning model for binary classification.

    Args:
        model_name: Backbone model name.
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained weights.

    Raises:
        ValueError: If unsupported model name is provided.
    """

    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        if model_name.lower() != "densenet121":
            raise ValueError(f"Unsupported model '{model_name}'. Only densenet121 is supported.")

        weights: DenseNet121_Weights | None = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone: nn.Module = densenet121(weights=weights)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )
        self.freeze_base_layers()

    def freeze_base_layers(self) -> None:
        """Freeze all backbone feature extractor parameters.

        Args:
            None.

        Returns:
            None.
        """
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_layers(self, n: int) -> None:
        """Unfreeze the last n denseblocks for fine-tuning.

        Args:
            n: Number of trailing dense blocks to unfreeze.

        Returns:
            None.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError("n must be non-negative.")
        block_names: list[str] = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
        selected_blocks: list[str] = block_names[-n:] if n > 0 else []
        for block_name in selected_blocks:
            block_module: nn.Module = getattr(self.backbone.features, block_name)
            for parameter in block_module.parameters():
                parameter.requires_grad = True

    def forward(self, inputs: Tensor) -> Tensor:
        """Run forward pass and return logits.

        Args:
            inputs: Input tensor with shape [B, C, H, W].

        Returns:
            Logits tensor with shape [B, num_classes].
        """
        return self.backbone(inputs)


class Trainer:
    """Training utility handling optimization, evaluation, and checkpoints.

    Args:
        model: Initialized model instance.
        config: Global configuration dictionary.
        device: Torch device for execution.
        class_weights: Class weight tensor for CrossEntropyLoss.
        logger: Logger for progress and metrics.
    """

    def __init__(
        self,
        model: ChestXRayModel,
        config: dict[str, Any],
        device: torch.device,
        class_weights: Tensor,
        logger: logging.Logger,
    ) -> None:
        self.model: ChestXRayModel = model.to(device)
        self.config: dict[str, Any] = config
        self.device: torch.device = device
        self.logger: logging.Logger = logger
        self.criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer: Optimizer = self._build_optimizer(float(config["training"]["learning_rate"]))
        self.scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=3,
            factor=0.5,
        )
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        )
        self.checkpoint_dir: Path = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self, learning_rate: float) -> Optimizer:
        """Create Adam optimizer using trainable parameters only.

        Args:
            learning_rate: Learning rate for optimizer.

        Returns:
            Configured Adam optimizer.
        """
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        return Adam(trainable_parameters, lr=learning_rate)

    def set_learning_rate(self, learning_rate: float) -> None:
        """Reinitialize optimizer and scheduler with a new learning rate.

        Args:
            learning_rate: New optimizer learning rate.

        Returns:
            None.
        """
        self.optimizer = self._build_optimizer(learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", patience=3, factor=0.5)
        self.logger.info("Optimizer reset with learning_rate=%.8f", learning_rate)

    def train_epoch(self, dataloader: DataLoader[tuple[Tensor, Tensor, list[str]]]) -> dict[str, float]:
        """Train model for one epoch.

        Args:
            dataloader: Training dataloader.

        Returns:
            Dictionary containing average loss and accuracy.
        """
        self.model.train()
        running_loss: float = 0.0
        correct_predictions: int = 0
        total_samples: int = 0

        progress_bar = tqdm(dataloader, desc="Train", leave=False)
        for batch_index, (images, labels, _) in enumerate(progress_bar, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits: Tensor = self.model(images)
                    loss: Tensor = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            predictions: Tensor = torch.argmax(logits, dim=1)
            batch_size: int = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            correct_predictions += int((predictions == labels).sum().item())
            total_samples += batch_size

            avg_loss: float = running_loss / max(total_samples, 1)
            avg_accuracy: float = correct_predictions / max(total_samples, 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_accuracy:.4f}")

            if batch_index % 50 == 0:
                self.logger.info(
                    "Training batch %d | avg_loss=%.4f | avg_accuracy=%.4f",
                    batch_index,
                    avg_loss,
                    avg_accuracy,
                )

        return {
            "avg_loss": running_loss / max(total_samples, 1),
            "accuracy": correct_predictions / max(total_samples, 1),
        }

    def evaluate(self, dataloader: DataLoader[tuple[Tensor, Tensor, list[str]]], split_name: str) -> dict[str, float]:
        """Evaluate model and compute classification metrics.

        Args:
            dataloader: Evaluation dataloader.
            split_name: Data split name for logging context.

        Returns:
            Dictionary containing loss, accuracy, precision, recall, f1, and auc_roc.
        """
        self.model.eval()
        total_loss: float = 0.0
        total_samples: int = 0
        all_labels: list[int] = []
        all_predictions: list[int] = []
        all_probabilities: list[float] = []

        with torch.no_grad():
            for images, labels, _ in dataloader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits: Tensor = self.model(images)
                loss: Tensor = self.criterion(logits, labels)
                probabilities: Tensor = torch.softmax(logits, dim=1)[:, 1]
                predictions: Tensor = torch.argmax(logits, dim=1)

                batch_size: int = labels.size(0)
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())
                all_probabilities.extend(probabilities.cpu().tolist())

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="binary",
            zero_division=0,
        )
        accuracy: float = float(np.mean(np.array(all_predictions) == np.array(all_labels)))
        try:
            auc_roc: float = float(roc_auc_score(all_labels, all_probabilities))
        except ValueError:
            auc_roc = float("nan")
            self.logger.warning(
                "AUC-ROC unavailable for split '%s' due to single-class ground truth in this evaluation batch.",
                split_name,
            )

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": auc_roc,
        }

    def save_checkpoint(self, epoch: int, metrics: dict[str, float], is_best: bool) -> None:
        """Save training checkpoint and optionally update best model.

        Args:
            epoch: Epoch number being saved.
            metrics: Metrics dictionary for the saved epoch.
            is_best: Whether this checkpoint is the current best model.

        Returns:
            None.

        Raises:
            OSError: If checkpoint cannot be written.
        """
        checkpoint_payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        checkpoint_path: Path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint_payload, checkpoint_path)
        if is_best:
            best_path: Path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint_payload, best_path)
            self.logger.info("Saved new best model checkpoint to %s", best_path)

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load a checkpoint and restore model and optimizer states.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Restored epoch number from checkpoint.

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
            RuntimeError: If checkpoint file cannot be loaded.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_data: dict[str, Any] = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        restored_epoch: int = int(checkpoint_data["epoch"])
        self.logger.info("Restored checkpoint from epoch %d (%s)", restored_epoch, checkpoint_path)
        return restored_epoch


def _load_preprocessing_module() -> Any:
    """Load preprocessing module from src/02_preprocessing.py dynamically.

    Args:
        None.

    Returns:
        Imported preprocessing module object.

    Raises:
        FileNotFoundError: If preprocessing script is missing.
        ImportError: If module loading fails.
    """
    preprocessing_path: Path = Path(__file__).resolve().parents[0] / "02_preprocessing.py"
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Preprocessing module not found: {preprocessing_path}")
    spec = importlib.util.spec_from_file_location("preprocessing_module", preprocessing_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to build import specification for preprocessing module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_training(
    config: dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, float], Path, float]:
    """Run two-phase training workflow and final test evaluation.

    Args:
        config: Global project configuration dictionary.
        device: Torch device used for training and evaluation.
        logger: Logger for training progress.

    Returns:
        Tuple of history dict, final test metrics, best model path, and total duration seconds.

    Raises:
        RuntimeError: If training prerequisites fail.
    """
    training_start: float = time.time()
    preprocessing_module: Any = _load_preprocessing_module()
    train_loader, val_loader, test_loader = preprocessing_module.get_dataloaders(config, logger)
    class_weights: Tensor = preprocessing_module.get_class_weights(train_loader.dataset)
    logger.info("Using class weights [NORMAL, PNEUMONIA]: %s", class_weights.tolist())

    model = ChestXRayModel(
        model_name=str(config["model"]["name"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=bool(config["model"]["pretrained"]),
    )
    trainer = Trainer(model=model, config=config, device=device, class_weights=class_weights, logger=logger)

    total_epochs: int = int(config["training"]["epochs"])
    if total_epochs <= 0:
        raise RuntimeError("Training epochs must be greater than zero.")
    phase1_epochs: int = max(1, total_epochs // 2)
    phase2_start_epoch: int = phase1_epochs + 1
    base_learning_rate: float = float(config["training"]["learning_rate"])
    early_stopping_patience: int = int(config["training"]["early_stopping_patience"])

    history: dict[str, list[dict[str, float]]] = {"epochs": []}
    best_val_f1: float = -1.0
    epochs_without_improvement: int = 0

    for epoch in range(1, total_epochs + 1):
        if epoch == phase2_start_epoch:
            model.unfreeze_layers(2)
            trainer.set_learning_rate(base_learning_rate / 10.0)
            logger.info("Phase 2 started at epoch %d: unfroze last 2 dense blocks.", epoch)

        current_phase: str = "feature_extraction" if epoch <= phase1_epochs else "fine_tuning"
        train_metrics: dict[str, float] = trainer.train_epoch(train_loader)
        val_metrics: dict[str, float] = trainer.evaluate(val_loader, split_name="val")
        val_f1_for_scheduler: float = val_metrics["f1"] if not np.isnan(val_metrics["f1"]) else 0.0
        trainer.scheduler.step(val_f1_for_scheduler)

        is_best: bool = val_metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_metrics: dict[str, float] = {
            "epoch": float(epoch),
            "phase_index": float(1 if current_phase == "feature_extraction" else 2),
            "train_loss": train_metrics["avg_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc_roc": val_metrics["auc_roc"],
            "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
        }
        history["epochs"].append(epoch_metrics)
        trainer.save_checkpoint(epoch=epoch, metrics=epoch_metrics, is_best=is_best)

        logger.info(
            "Phase=%s | Epoch=%d/%d | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f "
            "| val_f1=%.4f | val_auc=%.4f | lr=%.8f",
            current_phase,
            epoch,
            total_epochs,
            train_metrics["avg_loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["auc_roc"],
            float(trainer.optimizer.param_groups[0]["lr"]),
        )

        if epochs_without_improvement >= early_stopping_patience:
            logger.info(
                "Early stopping triggered after %d epochs without val F1 improvement.",
                epochs_without_improvement,
            )
            break

    best_model_path: Path = trainer.checkpoint_dir / "best_model.pth"
    trainer.load_checkpoint(best_model_path)
    test_metrics: dict[str, float] = trainer.evaluate(test_loader, split_name="test")

    logger.info("==============================================================")
    logger.info("Final test metrics | loss=%.4f", test_metrics["loss"])
    logger.info("Final test metrics | acc=%.4f", test_metrics["accuracy"])
    logger.info("Final test metrics | precision=%.4f", test_metrics["precision"])
    logger.info("Final test metrics | recall=%.4f", test_metrics["recall"])
    logger.info("Final test metrics | f1=%.4f", test_metrics["f1"])
    logger.info("Final test metrics | auc_roc=%.4f", test_metrics["auc_roc"])
    logger.info("==============================================================")

    training_duration_seconds: float = time.time() - training_start
    return history, test_metrics, best_model_path, training_duration_seconds


def plot_training_history(history: dict[str, list[dict[str, float]]], phase2_start_epoch: int) -> Path:
    """Plot and save training history visualizations.

    Args:
        history: History dictionary with per-epoch metrics.
        phase2_start_epoch: Epoch where fine-tuning phase starts.

    Returns:
        Output path to saved training history figure.

    Raises:
        ValueError: If history has no epochs.
        OSError: If figure cannot be written.
    """
    epoch_entries: list[dict[str, float]] = history.get("epochs", [])
    if not epoch_entries:
        raise ValueError("History contains no epoch data to plot.")

    epochs: list[int] = [int(entry["epoch"]) for entry in epoch_entries]
    train_loss: list[float] = [entry["train_loss"] for entry in epoch_entries]
    val_loss: list[float] = [entry["val_loss"] for entry in epoch_entries]
    train_acc: list[float] = [entry["train_accuracy"] for entry in epoch_entries]
    val_acc: list[float] = [entry["val_accuracy"] for entry in epoch_entries]
    val_f1: list[float] = [entry["val_f1"] for entry in epoch_entries]
    val_auc: list[float] = [entry["val_auc_roc"] for entry in epoch_entries]

    best_epoch_idx: int = int(np.nanargmax(np.array(val_f1)))
    best_epoch: int = epochs[best_epoch_idx]
    best_f1: float = val_f1[best_epoch_idx]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    subplot_titles: list[str] = [
        "Train vs Val Loss",
        "Train vs Val Accuracy",
        "Validation F1 Score",
        "Validation AUC-ROC",
    ]
    for axis, title in zip(axes.flatten(), subplot_titles):
        axis.set_title(title)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.axvline(phase2_start_epoch, color="gray", linestyle="--", linewidth=1, label="Phase 2 start")

    axes[0, 0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0, 0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, train_acc, marker="o", label="Train Accuracy")
    axes[0, 1].plot(epochs, val_acc, marker="o", label="Val Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, val_f1, marker="o", color="tab:green", label="Val F1")
    axes[1, 0].plot(best_epoch, best_f1, marker="*", markersize=14, color="red", label=f"Best Epoch {best_epoch}")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, val_auc, marker="o", color="tab:purple", label="Val AUC-ROC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC-ROC")
    axes[1, 1].legend()

    figure.suptitle("Chest X-Ray Training History", fontsize=16)
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    output_path: Path = Path(__file__).resolve().parents[1] / "reports" / "figures" / "training_history.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def _make_json_safe(value: Any) -> Any:
    """Convert NaN floats to None for JSON serialization.

    Args:
        value: Any nested JSON-like value.

    Returns:
        JSON-safe structure with NaNs replaced by None.
    """
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    return value


def save_training_results(
    history: dict[str, list[dict[str, float]]],
    test_metrics: dict[str, float],
    training_duration_seconds: float,
    config: dict[str, Any],
    device: torch.device,
) -> Path:
    """Save training outputs and metadata to JSON report.

    Args:
        history: Per-epoch training history.
        test_metrics: Final test metrics.
        training_duration_seconds: End-to-end training duration.
        config: Global configuration dictionary.
        device: Device used during training.

    Returns:
        Path to saved training results JSON file.

    Raises:
        OSError: If output file cannot be written.
    """
    results_payload: dict[str, Any] = {
        "per_epoch_history": _make_json_safe(history),
        "test_metrics": _make_json_safe(test_metrics),
        "training_duration_seconds": round(training_duration_seconds, 3),
        "model_config": {
            "name": config["model"]["name"],
            "num_classes": config["model"]["num_classes"],
            "pretrained": config["model"]["pretrained"],
        },
        "device_used": str(device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    output_path: Path = Path(__file__).resolve().parents[1] / "reports" / "training_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(results_payload, file_pointer, indent=2)
    return output_path


def main() -> None:
    """Entrypoint for standalone model training execution.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: If training pipeline execution fails.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    project_root: Path = Path(__file__).resolve().parents[1]
    logger: logging.Logger = setup_logging(project_root / "reports" / "training.log")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Detected device: %s", device)
    logger.info("GPU recommended. CPU training may take 2-4 hours for full run.")

    try:
        config: dict[str, Any] = load_config(project_root / "configs" / "config.yaml")
        history, test_metrics, best_model_path, training_duration_seconds = run_training(config, device, logger)
        total_epochs: int = int(config["training"]["epochs"])
        phase2_start_epoch: int = max(1, total_epochs // 2) + 1
        history_plot_path: Path = plot_training_history(history, phase2_start_epoch)
        results_json_path: Path = save_training_results(
            history=history,
            test_metrics=test_metrics,
            training_duration_seconds=training_duration_seconds,
            config=config,
            device=device,
        )
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        ValueError,
        RuntimeError,
        OSError,
        yaml.YAMLError,
        torch.cuda.CudaError,
    ) as exc:
        raise RuntimeError(f"Training pipeline failed: {exc}") from exc

    logger.info("Saved training history plot to %s", history_plot_path)
    logger.info("Saved training results JSON to %s", results_json_path)
    logger.info("Training completed successfully. Best model path: %s", best_model_path)


if __name__ == "__main__":
    main()
