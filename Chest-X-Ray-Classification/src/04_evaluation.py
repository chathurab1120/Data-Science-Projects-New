"""Comprehensive evaluation pipeline for chest X-ray classification models."""

from __future__ import annotations

import importlib.util
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

CLASS_NAMES: dict[int, str] = {0: "NORMAL", 1: "PNEUMONIA"}


def setup_logging(log_file_path: Path) -> logging.Logger:
    """Configure console and file logging.

    Args:
        log_file_path: Path to the output log file.

    Returns:
        Configured logger instance.

    Raises:
        OSError: If log file directory cannot be created.
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = logging.getLogger("evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler: logging.FileHandler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from disk.

    Args:
        config_path: Path to configuration file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config is malformed.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as file_pointer:
            config_data: Any = yaml.safe_load(file_pointer)
    except yaml.YAMLError:
        raise
    except OSError as exc:
        raise ValueError(f"Failed to read config: {config_path}") from exc
    if not isinstance(config_data, dict):
        raise ValueError("Config content must be a dictionary.")
    return config_data


def _load_script_module(module_name: str, script_path: Path) -> Any:
    """Load a Python script module by path.

    Args:
        module_name: Module name to assign.
        script_path: Absolute path to .py file.

    Returns:
        Loaded module object.

    Raises:
        FileNotFoundError: If module file does not exist.
        ImportError: If import fails.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Module script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_best_model(config: dict[str, Any], device: torch.device, logger: logging.Logger) -> nn.Module:
    """Load best trained model checkpoint for evaluation.

    Args:
        config: Project configuration dictionary.
        device: Torch device for model inference.
        logger: Logger for status messages.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If checkpoint file is missing.
        KeyError: If checkpoint data is missing required keys.
    """
    project_root: Path = Path(__file__).resolve().parents[1]
    training_module = _load_script_module("model_training", project_root / "src" / "03_model_training.py")
    model_class: type[nn.Module] = training_module.ChestXRayModel

    model: nn.Module = model_class(
        model_name=str(config["model"]["name"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=bool(config["model"]["pretrained"]),
    )
    checkpoint_path: Path = project_root / "models" / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    checkpoint_data: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)
    model.eval()
    epoch_value: int = int(checkpoint_data.get("epoch", -1))
    val_f1_value: float = float(checkpoint_data.get("metrics", {}).get("val_f1", float("nan")))
    logger.info("Loaded best model from %s | epoch=%d | val_f1=%.4f", checkpoint_path, epoch_value, val_f1_value)
    return model


def evaluate_full_test_set(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor, list[str]]],
    device: torch.device,
) -> dict[str, Any]:
    """Run inference on full test set and compute metrics.

    Args:
        model: Trained model in evaluation mode.
        dataloader: Test dataloader.
        device: Torch device.

    Returns:
        Dictionary containing predictions, probabilities, labels, paths, and metrics.
    """
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    all_paths: list[str] = []

    model.eval()
    with torch.no_grad():
        for images, labels, image_paths in tqdm(dataloader, desc="Evaluating test set", leave=False):
            images = images.to(device, non_blocking=True)
            logits: Tensor = model(images)
            probabilities: Tensor = torch.softmax(logits, dim=1)[:, 1]
            predictions: Tensor = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predictions.cpu().tolist())
            all_probs.extend(probabilities.cpu().tolist())
            all_paths.extend(list(image_paths))

    accuracy: float = float(accuracy_score(all_labels, all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    auc_roc_value: float = float(roc_auc_score(all_labels, all_probs))
    matrix: np.ndarray = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn: int = int(matrix[0, 0])
    fp: int = int(matrix[0, 1])
    specificity: float = float(tn / max(tn + fp, 1))

    metrics_payload: dict[str, float] = {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": auc_roc_value,
        "specificity": specificity,
    }
    return {
        "all_preds": all_preds,
        "all_probs": all_probs,
        "all_labels": all_labels,
        "all_paths": all_paths,
        "metrics": metrics_payload,
    }


def plot_confusion_matrix(labels: list[int], preds: list[int], output_path: Path) -> None:
    """Plot confusion matrix with counts and percentages.

    Args:
        labels: Ground-truth labels.
        preds: Predicted labels.
        output_path: Output figure path.

    Returns:
        None.
    """
    matrix: np.ndarray = confusion_matrix(labels, preds, labels=[0, 1])
    percentages: np.ndarray = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    annotations: np.ndarray = np.empty_like(matrix).astype(object)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            annotations[row_index, column_index] = (
                f"{matrix[row_index, column_index]}\n({percentages[row_index, column_index] * 100:.1f}%)"
            )

    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        ax=axis,
    )
    axis.set_title("Confusion Matrix — Test Set (n=624)")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_roc_curve(labels: list[int], probs: list[float], output_path: Path) -> None:
    """Plot ROC curve with AUC and threshold-0.5 operating point.

    Args:
        labels: Ground-truth labels.
        probs: Probability scores for PNEUMONIA class.
        output_path: Output figure path.

    Returns:
        None.
    """
    fpr_values, tpr_values, thresholds = roc_curve(labels, probs)
    auc_score: float = float(roc_auc_score(labels, probs))

    threshold_array: np.ndarray = np.asarray(thresholds)
    closest_index: int = int(np.argmin(np.abs(threshold_array - 0.5)))
    op_fpr: float = float(fpr_values[closest_index])
    op_tpr: float = float(tpr_values[closest_index])

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(fpr_values, tpr_values, linewidth=2.2, label=f"ROC (AUC = {auc_score:.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    axis.scatter(op_fpr, op_tpr, color="red", s=55, label="Threshold = 0.5")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve — Test Set")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_precision_recall_curve(labels: list[int], probs: list[float], output_path: Path) -> None:
    """Plot precision-recall curve with baseline and AUC-PR.

    Args:
        labels: Ground-truth labels.
        probs: Probability scores for PNEUMONIA class.
        output_path: Output figure path.

    Returns:
        None.
    """
    precision_values, recall_values, _ = precision_recall_curve(labels, probs)
    auc_pr: float = float(auc(recall_values, precision_values))
    prevalence: float = float(np.mean(np.asarray(labels)))

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(recall_values, precision_values, linewidth=2.2, label=f"PR Curve (AUC = {auc_pr:.4f})")
    axis.hlines(y=prevalence, xmin=0.0, xmax=1.0, linestyle="--", color="gray", label=f"Baseline = {prevalence:.3f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curve — Test Set")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="lower left")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


class GradCAM:
    """Grad-CAM generator for model interpretability.

    Args:
        model: Trained model instance.
        target_layer_name: Layer name for Grad-CAM hooks.
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "features.denseblock4") -> None:
        self.model: nn.Module = model
        self.target_layer_name: str = target_layer_name
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self.forward_handle: Any = None
        self.backward_handle: Any = None
        self._register_hooks()

    def _resolve_target_layer(self) -> nn.Module:
        """Resolve target layer reference from model named modules.

        Args:
            None.

        Returns:
            Target module for hook registration.

        Raises:
            ValueError: If target layer cannot be found.
        """
        named_modules: dict[str, nn.Module] = dict(self.model.named_modules())
        candidate_names: list[str] = [self.target_layer_name, f"backbone.{self.target_layer_name}"]
        for candidate_name in candidate_names:
            if candidate_name in named_modules:
                return named_modules[candidate_name]
        raise ValueError(f"Target layer '{self.target_layer_name}' not found in model modules.")

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer.

        Args:
            None.

        Returns:
            None.
        """
        target_layer: nn.Module = self._resolve_target_layer()
        for parameter in target_layer.parameters():
            parameter.requires_grad = True

        def forward_hook(_: nn.Module, __: tuple[Tensor, ...], output: Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(_: nn.Module, __: tuple[Tensor, ...], grad_output: tuple[Tensor, ...]) -> None:
            self.gradients = grad_output[0].detach()

        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)

    def compute(self, input_tensor: Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single input tensor.

        Args:
            input_tensor: Input tensor of shape [1, C, H, W].

        Returns:
            Normalized heatmap array of shape [224, 224].

        Raises:
            RuntimeError: If hooks fail to capture activations/gradients.
        """
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits: Tensor = self.model(input_tensor)
        predicted_index: int = int(torch.argmax(logits, dim=1).item())
        class_score: Tensor = logits[:, predicted_index]
        class_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture required tensors.")

        pooled_gradients: Tensor = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_activations: Tensor = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)
        heatmap: Tensor = torch.relu(weighted_activations)
        heatmap = torch.nn.functional.interpolate(heatmap, size=(224, 224), mode="bilinear", align_corners=False)
        heatmap_array: np.ndarray = heatmap.squeeze().detach().cpu().numpy()

        max_value: float = float(np.max(heatmap_array))
        min_value: float = float(np.min(heatmap_array))
        if max_value > min_value:
            heatmap_array = (heatmap_array - min_value) / (max_value - min_value)
        else:
            heatmap_array = np.zeros((224, 224), dtype=np.float32)
        return heatmap_array.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove registered hooks to avoid memory leaks.

        Args:
            None.

        Returns:
            None.
        """
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None


def _denormalize_image(image_tensor: Tensor, mean_values: list[float], std_values: list[float]) -> np.ndarray:
    """Convert normalized tensor image back to RGB numpy format.

    Args:
        image_tensor: Image tensor [C, H, W].
        mean_values: Normalization means.
        std_values: Normalization stds.

    Returns:
        RGB uint8 image array [H, W, 3].
    """
    image_array: np.ndarray = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean_array: np.ndarray = np.asarray(mean_values, dtype=np.float32).reshape(1, 1, 3)
    std_array: np.ndarray = np.asarray(std_values, dtype=np.float32).reshape(1, 1, 3)
    image_array = (image_array * std_array) + mean_array
    image_array = np.clip(image_array, 0.0, 1.0)
    return (image_array * 255.0).astype(np.uint8)


def _select_correct_indices(model: nn.Module, dataset: Any, device: torch.device, n: int) -> list[int]:
    """Select balanced correct prediction indices for Grad-CAM examples.

    Args:
        model: Trained model.
        dataset: Dataset object with __getitem__.
        device: Torch device.
        n: Total number of examples to select.

    Returns:
        Selected dataset indices.
    """
    required_per_class: int = n // 2
    selected_indices: dict[int, list[int]] = {0: [], 1: []}
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            image_tensor, true_label, _ = dataset[index]
            input_batch: Tensor = image_tensor.unsqueeze(0).to(device)
            prediction: int = int(torch.argmax(model(input_batch), dim=1).item())
            if prediction == true_label and len(selected_indices[true_label]) < required_per_class:
                selected_indices[true_label].append(index)
            if all(len(indices) >= required_per_class for indices in selected_indices.values()):
                break
    return selected_indices[0] + selected_indices[1]


def plot_gradcam_examples(model: nn.Module, dataset: Any, device: torch.device, n: int = 8) -> Path:
    """Generate Grad-CAM overlays for balanced correct examples.

    Args:
        model: Trained model.
        dataset: Test dataset.
        device: Torch device.
        n: Number of examples to visualize.

    Returns:
        Path to saved figure.
    """
    project_root: Path = Path(__file__).resolve().parents[1]
    config: dict[str, Any] = load_config(project_root / "configs" / "config.yaml")
    mean_values: list[float] = [float(value) for value in config["image"]["normalize"]["mean"]]
    std_values: list[float] = [float(value) for value in config["image"]["normalize"]["std"]]
    selected_indices: list[int] = _select_correct_indices(model, dataset, device, n=n)
    gradcam_generator = GradCAM(model)

    figure, axes = plt.subplots(len(selected_indices), 2, figsize=(10, 4 * len(selected_indices)))
    if len(selected_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, sample_index in enumerate(selected_indices):
        image_tensor, true_label, image_path = dataset[sample_index]
        input_batch: Tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits: Tensor = model(input_batch)
            probabilities: Tensor = torch.softmax(logits, dim=1)
        predicted_label: int = int(torch.argmax(probabilities, dim=1).item())
        predicted_probability: float = float(probabilities[0, predicted_label].item())

        heatmap: np.ndarray = gradcam_generator.compute(input_batch)
        original_image: np.ndarray = _denormalize_image(image_tensor, mean_values, std_values)
        heatmap_uint8: np.ndarray = np.uint8(255 * heatmap)
        colormap_bgr: np.ndarray = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb: np.ndarray = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)
        overlay_image: np.ndarray = cv2.addWeighted(original_image, 0.5, colormap_rgb, 0.5, 0)

        axes[row_index, 0].imshow(original_image)
        axes[row_index, 0].set_title(f"Original\n{Path(image_path).name}", fontsize=10)
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(overlay_image)
        axes[row_index, 1].set_title(
            f"True: {CLASS_NAMES[true_label]} | Pred: {CLASS_NAMES[predicted_label]} | Prob: {predicted_probability*100:.1f}%",
            fontsize=10,
        )
        axes[row_index, 1].axis("off")

    gradcam_generator.remove_hooks()
    figure.suptitle("Grad-CAM Examples — Correct Test Predictions", fontsize=14)
    figure.tight_layout(rect=(0, 0.01, 1, 0.98))
    output_path: Path = project_root / "reports" / "figures" / "gradcam_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return output_path


def plot_misclassified(model: nn.Module, dataset: Any, device: torch.device, n: int = 8) -> Path:
    """Visualize highest-confidence misclassified test samples.

    Args:
        model: Trained model.
        dataset: Test dataset.
        device: Torch device.
        n: Number of misclassified examples to show.

    Returns:
        Path to saved misclassification figure.
    """
    project_root: Path = Path(__file__).resolve().parents[1]
    config: dict[str, Any] = load_config(project_root / "configs" / "config.yaml")
    mean_values: list[float] = [float(value) for value in config["image"]["normalize"]["mean"]]
    std_values: list[float] = [float(value) for value in config["image"]["normalize"]["std"]]
    errors: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            image_tensor, true_label, image_path = dataset[index]
            input_batch: Tensor = image_tensor.unsqueeze(0).to(device)
            logits: Tensor = model(input_batch)
            probabilities: Tensor = torch.softmax(logits, dim=1)
            predicted_label: int = int(torch.argmax(probabilities, dim=1).item())
            confidence: float = float(probabilities[0, predicted_label].item())
            if predicted_label != true_label:
                errors.append(
                    {
                        "index": index,
                        "confidence": confidence,
                        "true_label": true_label,
                        "pred_label": predicted_label,
                        "path": image_path,
                        "image_tensor": image_tensor,
                    }
                )

    selected_errors: list[dict[str, Any]] = sorted(errors, key=lambda item: item["confidence"], reverse=True)[:n]
    rows: int = 2
    cols: int = 4
    figure, axes = plt.subplots(rows, cols, figsize=(16, 8))

    for axis_index, axis in enumerate(axes.flat):
        if axis_index >= len(selected_errors):
            axis.axis("off")
            continue
        error_item: dict[str, Any] = selected_errors[axis_index]
        display_image: np.ndarray = _denormalize_image(error_item["image_tensor"], mean_values, std_values)
        axis.imshow(display_image)
        axis.set_title(
            f"True: {CLASS_NAMES[int(error_item['true_label'])]} | "
            f"Pred: {CLASS_NAMES[int(error_item['pred_label'])]} | "
            f"Conf: {float(error_item['confidence'])*100:.1f}%",
            fontsize=9,
        )
        axis.axis("off")

    figure.suptitle("Highest-Confidence Misclassified Test Examples", fontsize=14)
    figure.tight_layout(rect=(0, 0.01, 1, 0.96))
    output_path: Path = project_root / "reports" / "figures" / "misclassified_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return output_path


def generate_classification_report(
    metrics: dict[str, float],
    labels: list[int],
    preds: list[int],
    probs: list[float],
    model_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Generate sklearn classification report and save evaluation JSON.

    Args:
        metrics: Main evaluation metrics dictionary.
        labels: Ground-truth labels.
        preds: Predicted labels at threshold 0.5.
        probs: Predicted probabilities for class 1.
        model_path: Path to model checkpoint used for evaluation.
        output_path: Path to save JSON report.
        logger: Logger for report content.

    Returns:
        None.
    """
    report_dict: dict[str, Any] = classification_report(
        labels,
        preds,
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    report_text: str = classification_report(
        labels,
        preds,
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        zero_division=0,
    )
    logger.info("Classification report:\n%s", report_text)

    threshold_values: list[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_metrics: dict[str, dict[str, float]] = {}
    labels_array: np.ndarray = np.asarray(labels)
    probs_array: np.ndarray = np.asarray(probs)
    for threshold in threshold_values:
        threshold_preds: np.ndarray = (probs_array >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_array, threshold_preds, average="binary", zero_division=0
        )
        threshold_matrix: np.ndarray = confusion_matrix(labels_array, threshold_preds, labels=[0, 1])
        tn: int = int(threshold_matrix[0, 0])
        fp: int = int(threshold_matrix[0, 1])
        specificity: float = float(tn / max(tn + fp, 1))
        threshold_metrics[f"{threshold:.1f}"] = {
            "accuracy": float(accuracy_score(labels_array, threshold_preds)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": specificity,
        }

    payload: dict[str, Any] = {
        "metrics": metrics,
        "classification_report": report_dict,
        "threshold_analysis": threshold_metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(payload, file_pointer, indent=2)


def main() -> None:
    """Run full evaluation workflow and artifact generation.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: If evaluation fails.
    """
    project_root: Path = Path(__file__).resolve().parents[1]
    logger: logging.Logger = setup_logging(project_root / "reports" / "evaluation.log")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        config: dict[str, Any] = load_config(project_root / "configs" / "config.yaml")
        model: nn.Module = load_best_model(config, device, logger)

        preprocessing_module = _load_script_module("preprocessing_module", project_root / "src" / "02_preprocessing.py")
        train_loader, val_loader, test_loader = preprocessing_module.get_dataloaders(config, logger)
        _ = preprocessing_module.get_class_weights(train_loader.dataset)
        del train_loader, val_loader

        evaluation_results: dict[str, Any] = evaluate_full_test_set(model, test_loader, device)
        labels: list[int] = evaluation_results["all_labels"]
        preds: list[int] = evaluation_results["all_preds"]
        probs: list[float] = evaluation_results["all_probs"]
        metrics: dict[str, float] = evaluation_results["metrics"]
        logger.info("Test metrics summary: %s", metrics)

        figures_dir: Path = project_root / "reports" / "figures"
        plot_confusion_matrix(labels, preds, figures_dir / "confusion_matrix.png")
        plot_roc_curve(labels, probs, figures_dir / "roc_curve.png")
        plot_precision_recall_curve(labels, probs, figures_dir / "precision_recall_curve.png")
        plot_gradcam_examples(model, test_loader.dataset, device, n=8)
        plot_misclassified(model, test_loader.dataset, device, n=8)

        generate_classification_report(
            metrics=metrics,
            labels=labels,
            preds=preds,
            probs=probs,
            model_path=project_root / "models" / "checkpoints" / "best_model.pth",
            output_path=project_root / "reports" / "evaluation_report.json",
            logger=logger,
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
        raise RuntimeError(f"Evaluation failed: {exc}") from exc

    logger.info("Evaluation complete. All figures saved to reports/figures/")


if __name__ == "__main__":
    main()
