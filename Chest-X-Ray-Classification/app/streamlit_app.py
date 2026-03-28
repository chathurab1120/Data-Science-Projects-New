"""Multi-page Streamlit dashboard for chest X-ray pneumonia classification."""

from __future__ import annotations

import importlib.util
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
import torch
import yaml
from PIL import Image

st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGGER: logging.Logger = logging.getLogger("streamlit_app")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.propagate = False

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def _load_script_module(module_name: str, script_path: Path) -> Any:
    """Load a Python module from script path.

    Args:
        module_name: Name for imported module.
        script_path: Path to Python script.

    Returns:
        Loaded module object.

    Raises:
        FileNotFoundError: If script path does not exist.
        ImportError: If module cannot be imported.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inject_custom_css() -> None:
    """Inject custom CSS for visual polish.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(
        """
        <style>
        .metric-card {
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 12px 16px;
            background-color: #FAFAFA;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            margin-bottom: 8px;
        }
        .hero-title {
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1rem;
        }
        .diagnosis-box {
            border-radius: 10px;
            padding: 12px;
            font-weight: 600;
            font-size: 1.05rem;
            border: 1px solid #E5E7EB;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_json_file(file_path: Path) -> dict[str, Any] | None:
    """Load JSON file from disk with caching.

    Args:
        file_path: Path to JSON file.

    Returns:
        Parsed dictionary or None if unavailable.
    """
    if not file_path.exists():
        return None
    try:
        with file_path.open("r", encoding="utf-8") as file_pointer:
            loaded_data: Any = json.load(file_pointer)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded_data, dict):
        return None
    return loaded_data


@st.cache_data(show_spinner=False)
def load_config_file(config_path: Path) -> dict[str, Any] | None:
    """Load YAML config file from disk with caching.

    Args:
        config_path: Path to config YAML.

    Returns:
        Parsed config dictionary or None.
    """
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as file_pointer:
            config_data: Any = yaml.safe_load(file_pointer)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(config_data, dict):
        return None
    return config_data


def _safe_image(path: Path, caption: str) -> None:
    """Render image if it exists, otherwise show warning.

    Args:
        path: Path to image file.
        caption: Caption text.

    Returns:
        None.
    """
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {path}")


def _extract_device_name(model_inference: Any) -> str:
    """Extract device name from model inference object.

    Args:
        model_inference: Loaded inference object.

    Returns:
        Device name string.
    """
    device_attr: Any = getattr(model_inference, "device", None)
    if isinstance(device_attr, torch.device):
        return str(device_attr)
    return "unknown"


@st.cache_resource(show_spinner=False)
def load_model_inference() -> Any:
    """Load reusable model inference object once.

    Args:
        None.

    Returns:
        Model inference instance.

    Raises:
        RuntimeError: If model cannot be initialized.
    """
    model_path: Path = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"
    config_path: Path = PROJECT_ROOT / "configs" / "config.yaml"

    try:
        from app.gradio_app import ModelInference  # type: ignore

        return ModelInference(model_path=model_path, config_path=config_path)
    except (ImportError, ModuleNotFoundError):
        LOGGER.warning("Could not import ModelInference from gradio app. Falling back to local implementation.")

        model_module = _load_script_module("model_training_module_streamlit", PROJECT_ROOT / "src" / "03_model_training.py")
        eval_module = _load_script_module("evaluation_module_streamlit", PROJECT_ROOT / "src" / "04_evaluation.py")
        torchvision_module = __import__("torchvision", fromlist=["transforms"])
        transforms = getattr(torchvision_module, "transforms")
        cv2_module = __import__("cv2")
        numpy_module = __import__("numpy")
        pil_imagedraw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])

        class _FallbackModelInference:
            """Fallback inference class if Gradio module import is unavailable."""

            def __init__(self, model_path: Path, config_path: Path) -> None:
                """Initialize fallback inference stack.

                Args:
                    model_path: Path to trained checkpoint.
                    config_path: Path to YAML configuration.

                Returns:
                    None.

                Raises:
                    RuntimeError: If config cannot be loaded.
                    KeyError: If expected checkpoint keys are missing.
                """
                self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                config_data: dict[str, Any] | None = load_config_file(config_path)
                if config_data is None:
                    raise RuntimeError("Unable to load config for fallback inference.")
                self.config: dict[str, Any] = config_data
                model_cls = model_module.ChestXRayModel
                self.model = model_cls(
                    model_name=str(self.config["model"]["name"]),
                    num_classes=int(self.config["model"]["num_classes"]),
                    pretrained=bool(self.config["model"]["pretrained"]),
                )
                checkpoint_data = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint_data["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4693, 0.4693, 0.4693], [0.2270, 0.2270, 0.2270]),
                    ]
                )
                self.gradcam_cls = eval_module.GradCAM

            def predict(self, image: Image.Image) -> dict[str, float | str]:
                """Run single-image inference.

                Args:
                    image: Input PIL chest X-ray image.

                Returns:
                    Prediction dictionary with class and probabilities.
                """
                rgb_image: Image.Image = image.convert("RGB")
                tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1)
                normal_prob: float = float(probs[0, 0].item())
                pneumonia_prob: float = float(probs[0, 1].item())
                pred_idx: int = int(torch.argmax(probs, dim=1).item())
                pred_class: str = "PNEUMONIA" if pred_idx == 1 else "NORMAL"
                return {
                    "predicted_class": pred_class,
                    "confidence": float(probs[0, pred_idx].item()),
                    "normal_prob": normal_prob,
                    "pneumonia_prob": pneumonia_prob,
                }

            def generate_gradcam(self, image: Image.Image) -> Image.Image:
                """Generate side-by-side Grad-CAM visualization.

                Args:
                    image: Input PIL chest X-ray image.

                Returns:
                    Side-by-side PIL image with original and attention overlay.
                """
                rgb_image: Image.Image = image.convert("RGB").resize((224, 224))
                tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
                gradcam = self.gradcam_cls(self.model, target_layer_name="features.denseblock4")
                try:
                    heatmap = gradcam.compute(tensor)
                finally:
                    gradcam.remove_hooks()
                original_array = numpy_module.asarray(rgb_image, dtype=numpy_module.uint8)
                heatmap_uint8 = numpy_module.uint8(255 * heatmap)
                heatmap_color = cv2_module.applyColorMap(heatmap_uint8, cv2_module.COLORMAP_JET)
                heatmap_rgb = cv2_module.cvtColor(heatmap_color, cv2_module.COLOR_BGR2RGB)
                overlay = cv2_module.addWeighted(original_array, 0.5, heatmap_rgb, 0.5, 0)
                combined = Image.new("RGB", (448, 254), color=(255, 255, 255))
                combined.paste(Image.fromarray(original_array), (0, 30))
                combined.paste(Image.fromarray(overlay), (224, 30))
                draw = pil_imagedraw.ImageDraw.Draw(combined)
                draw.text((70, 8), "Original", fill=(0, 0, 0))
                draw.text((258, 8), "AI Focus Area", fill=(0, 0, 0))
                return combined

        return _FallbackModelInference(model_path=model_path, config_path=config_path)


def render_overview_page(data_summary: dict[str, Any] | None) -> None:
    """Render project overview page.

    Args:
        data_summary: Data summary JSON payload.

    Returns:
        None.
    """
    st.markdown('<div class="hero-title">🫁 Chest X-Ray Pneumonia Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Production-grade deep learning system for pneumonia detection</div>',
        unsafe_allow_html=True,
    )
    st.warning("⚠️ Research use only. This dashboard does not provide medical diagnosis.")

    metric_columns = st.columns(6)
    metric_columns[0].metric("Test Accuracy", "87.8%")
    metric_columns[1].metric("Precision", "84.0%")
    metric_columns[2].metric("Recall", "99.5%")
    metric_columns[3].metric("F1 Score", "91.1%")
    metric_columns[4].metric("AUC-ROC", "96.8%")
    metric_columns[5].metric("Specificity", "68.4%")

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("About the Project")
        if data_summary is None:
            st.warning("Missing `reports/data_summary.json`.")
        else:
            split_counts: dict[str, int] = data_summary.get("split_counts", {})
            imbalance_ratio: float = float(data_summary.get("training_imbalance_ratio", float("nan")))
            st.markdown(
                "This project detects pneumonia from chest X-rays using transfer learning on DenseNet121. "
                "The pipeline includes robust preprocessing, class-imbalance handling, explainability, and deployment."
            )
            st.markdown(
                f"- **Train**: {split_counts.get('train', 'N/A')} images\n"
                f"- **Val**: {split_counts.get('val', 'N/A')} images\n"
                f"- **Test**: {split_counts.get('test', 'N/A')} images\n"
                f"- **Class imbalance ratio (train)**: {imbalance_ratio:.4f}"
            )

    with right_column:
        st.subheader("Tech Stack")
        stack_table: list[dict[str, str]] = [
            {"Component": "Model", "Value": "DenseNet121 (Transfer Learning)"},
            {"Component": "Framework", "Value": "PyTorch"},
            {"Component": "Deployment", "Value": "Hugging Face Spaces + Streamlit"},
            {"Component": "Dataset", "Value": "Kaggle Chest X-Ray (Pneumonia)"},
            {"Component": "Python", "Value": "3.11"},
        ]
        st.table(stack_table)

    _safe_image(FIGURES_DIR / "sample_images.png", "Sample chest X-ray images from training data.")
    _safe_image(
        FIGURES_DIR / "class_distribution.png",
        "Class distribution across splits, highlighting training imbalance.",
    )


def render_performance_page(
    evaluation_report: dict[str, Any] | None,
    training_results: dict[str, Any] | None,
) -> None:
    """Render model performance page.

    Args:
        evaluation_report: Evaluation JSON payload.
        training_results: Training results JSON payload.

    Returns:
        None.
    """
    st.title("Model Performance & Evaluation")

    if evaluation_report is None:
        st.warning("Missing `reports/evaluation_report.json`.")
        return
    if training_results is None:
        st.warning("Missing `reports/training_results.json`.")
        return

    metrics: dict[str, float] = evaluation_report.get("metrics", {})
    top_columns = st.columns(4)
    top_columns[0].metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.2f}%", delta="↑ strong")
    top_columns[1].metric("Precision", f"{metrics.get('precision', 0.0) * 100:.2f}%", delta="↔ balanced")
    top_columns[2].metric("Recall", f"{metrics.get('recall', 0.0) * 100:.2f}%", delta="↑ critical")
    top_columns[3].metric("F1 Score", f"{metrics.get('f1', 0.0) * 100:.2f}%", delta="↑ robust")

    left_column, right_column = st.columns(2)
    with left_column:
        _safe_image(FIGURES_DIR / "confusion_matrix.png", "Confusion matrix on test set.")
        st.markdown(
            "**Medical context:**\n"
            "- **TP**: Pneumonia correctly detected\n"
            "- **FP**: Healthy case flagged as pneumonia\n"
            "- **TN**: Healthy case correctly identified\n"
            "- **FN**: Pneumonia case missed (most critical error)"
        )
    with right_column:
        _safe_image(FIGURES_DIR / "roc_curve.png", "ROC curve for test predictions.")
        st.markdown(
            "**AUC-ROC (plain English):**\n"
            "Higher values indicate better separation between NORMAL and PNEUMONIA across thresholds."
        )

    _safe_image(FIGURES_DIR / "precision_recall_curve.png", "Precision-Recall curve on test set.")

    st.subheader("Training History")
    _safe_image(FIGURES_DIR / "training_history.png", "Training and validation trends by epoch.")

    epoch_history: list[dict[str, Any]] = training_results.get("per_epoch_history", {}).get("epochs", [])
    history_rows: list[dict[str, Any]] = []
    for entry in epoch_history:
        phase_label: str = "Feature Extraction" if int(entry.get("phase_index", 1)) == 1 else "Fine-Tuning"
        history_rows.append(
            {
                "epoch": int(entry.get("epoch", 0)),
                "phase": phase_label,
                "train_loss": round(float(entry.get("train_loss", 0.0)), 4),
                "train_acc": round(float(entry.get("train_accuracy", 0.0)), 4),
                "val_loss": round(float(entry.get("val_loss", 0.0)), 4),
                "val_acc": round(float(entry.get("val_accuracy", 0.0)), 4),
                "val_f1": round(float(entry.get("val_f1", 0.0)), 4),
                "val_auc": round(float(entry.get("val_auc_roc", 0.0)), 4),
            }
        )
    st.dataframe(history_rows, use_container_width=True)

    st.subheader("Threshold Analysis")
    threshold_analysis: dict[str, dict[str, float]] = evaluation_report.get("threshold_analysis", {})
    threshold_rows: list[dict[str, Any]] = []
    for threshold_key, threshold_metrics in threshold_analysis.items():
        threshold_rows.append(
            {
                "threshold": float(threshold_key),
                "accuracy": float(threshold_metrics.get("accuracy", 0.0)),
                "precision": float(threshold_metrics.get("precision", 0.0)),
                "recall": float(threshold_metrics.get("recall", 0.0)),
                "f1": float(threshold_metrics.get("f1", 0.0)),
                "specificity": float(threshold_metrics.get("specificity", 0.0)),
            }
        )
    threshold_rows = sorted(threshold_rows, key=lambda row: row["threshold"])
    try:
        import pandas as pd

        threshold_df = pd.DataFrame(threshold_rows)

        def _style_threshold_row(row: Any) -> list[str]:
            """Highlight operating threshold row.

            Args:
                row: Dataframe row.

            Returns:
                CSS style list for row cells.
            """
            if abs(float(row["threshold"]) - 0.5) < 1e-9:
                return ["background-color: #FFF3CD; font-weight: 700;"] * len(row)
            return [""] * len(row)

        styled_df = threshold_df.style.apply(_style_threshold_row, axis=1).format({"threshold": "{:.1f}"})
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    except ModuleNotFoundError:
        st.dataframe(
            threshold_rows,
            use_container_width=True,
            hide_index=True,
            column_config={"threshold": st.column_config.NumberColumn(format="%.1f")},
        )
        st.caption("Install pandas to enable highlighted threshold row styling.")
    st.info("Threshold 0.5 is the current operating point; lower thresholds usually increase recall but reduce precision.")


def _build_example_options() -> dict[str, Path]:
    """Build mapping of example labels to dataset image paths.

    Args:
        None.

    Returns:
        Mapping from display labels to file paths.
    """
    base_test_dir: Path = PROJECT_ROOT / "data" / "raw" / "chest_xray" / "test"
    normal_paths: list[Path] = sorted((base_test_dir / "NORMAL").glob("*.jp*g"))
    pneumonia_paths: list[Path] = sorted((base_test_dir / "PNEUMONIA").glob("*.jp*g"))
    option_map: dict[str, Path] = {}
    if normal_paths:
        option_map["Example 1 - NORMAL"] = normal_paths[0]
    if len(pneumonia_paths) >= 1:
        option_map["Example 2 - PNEUMONIA"] = pneumonia_paths[0]
    if len(pneumonia_paths) >= 2:
        option_map["Example 3 - PNEUMONIA"] = pneumonia_paths[1]
    return option_map


def _build_assessment_text(pneumonia_prob: float, normal_prob: float) -> str:
    """Create risk assessment message for live inference.

    Args:
        pneumonia_prob: Pneumonia probability score.
        normal_prob: Normal probability score.

    Returns:
        Assessment message.
    """
    if pneumonia_prob > 0.70:
        core_text: str = "High likelihood of pneumonia detected. Please consult a radiologist immediately."
    elif 0.40 <= pneumonia_prob <= 0.70:
        core_text = "Moderate indicators present. Further clinical evaluation recommended."
    elif normal_prob > 0.70:
        core_text = "Low indicators of pneumonia. Routine monitoring advised."
    else:
        core_text = "Uncertain confidence. Further clinical evaluation recommended."
    return (
        f"{core_text}\n\n"
        "⚠️ Research use only.\n"
        "Not a substitute for professional medical diagnosis."
    )


def render_live_prediction_page(model_inference: Any | None) -> None:
    """Render live prediction page.

    Args:
        model_inference: Loaded model inference object.

    Returns:
        None.
    """
    st.title("Live X-Ray Analysis")
    st.warning("⚠️ This is a research tool only and not for clinical diagnosis.")

    left_column, right_column = st.columns(2)
    selected_image: Image.Image | None = None

    with left_column:
        uploaded_file = st.file_uploader("Upload chest X-ray image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                st.image(selected_image, caption="Uploaded X-ray", use_container_width=True)
            except (OSError, ValueError):
                st.warning("Could not read uploaded image.")

        example_options: dict[str, Path] = _build_example_options()
        if example_options:
            selected_example: str = st.selectbox("Use example image", options=["None"] + list(example_options.keys()))
            if selected_example != "None":
                example_path: Path = example_options[selected_example]
                try:
                    selected_image = Image.open(example_path).convert("RGB")
                    st.image(selected_image, caption=selected_example, use_container_width=True)
                except (OSError, ValueError):
                    st.warning(f"Could not load example image: {example_path}")

        analyze_clicked: bool = st.button("Analyze")

    with right_column:
        if analyze_clicked:
            if model_inference is None:
                st.error("Model is not loaded. Please check configuration and checkpoint files.")
            elif selected_image is None:
                st.warning("Please upload an image or select an example.")
            else:
                try:
                    with st.spinner("Analyzing X-Ray..."):
                        prediction: dict[str, Any] = model_inference.predict(selected_image)
                        gradcam_image: Image.Image = model_inference.generate_gradcam(selected_image)
                    predicted_class: str = str(prediction["predicted_class"])
                    pneumonia_prob: float = float(prediction["pneumonia_prob"])
                    normal_prob: float = float(prediction["normal_prob"])
                    if predicted_class == "PNEUMONIA":
                        st.error("🔴 PNEUMONIA DETECTED")
                    else:
                        st.success("✅ NORMAL")
                    probability_columns = st.columns(2)
                    probability_columns[0].metric("PNEUMONIA Probability", f"{pneumonia_prob*100:.1f}%")
                    probability_columns[1].metric("NORMAL Probability", f"{normal_prob*100:.1f}%")
                    st.info(_build_assessment_text(pneumonia_prob, normal_prob))
                    st.image(
                        gradcam_image,
                        caption="Grad-CAM overlay: red regions indicate stronger model attention.",
                        use_container_width=True,
                    )
                except (RuntimeError, ValueError, KeyError, OSError) as exc:
                    LOGGER.error("Prediction error: %s\n%s", exc, traceback.format_exc())
                    st.error("Inference failed. Please try another image.")


def render_explainability_page() -> None:
    """Render explainability page.

    Args:
        None.

    Returns:
        None.
    """
    st.title("Model Explainability — Grad-CAM Analysis")
    st.markdown(
        "Grad-CAM highlights image regions that most influence the model prediction. "
        "Explainability helps stakeholders validate whether the model focuses on clinically meaningful structures."
    )
    st.markdown(
        "- **Why explainability matters**: increases trust, reveals failure modes, and supports model governance.\n"
        "- **How to read heatmaps**: red indicates high attention; blue indicates low attention."
    )

    _safe_image(
        FIGURES_DIR / "gradcam_examples.png",
        "Grad-CAM examples across correctly classified NORMAL and PNEUMONIA test samples.",
    )
    _safe_image(
        FIGURES_DIR / "misclassified_examples.png",
        "Cases where the model made mistakes — understanding failure modes.",
    )

    with st.expander("Explainability insights"):
        st.markdown(
            "- In many **PNEUMONIA** cases, attention clusters around dense or diffuse opacities.\n"
            "- In **NORMAL** cases, attention is often more distributed with less focal pathology emphasis.\n"
            "- Failure modes include atypical anatomy, low contrast, artifacts, or severe overlap between classes.\n"
            "- This system prioritizes **high recall (99.5%)** to reduce missed pneumonia cases, accepting some false positives."
        )

    st.subheader("Model Architecture")
    st.code(
        "DenseNet121 Backbone\n"
        "  -> Classifier Head:\n"
        "       Linear(1024 -> 512)\n"
        "       ReLU()\n"
        "       Dropout(0.4)\n"
        "       Linear(512 -> 2)\n",
        language="text",
    )


def render_sidebar(model_inference: Any | None) -> str:
    """Render sidebar and return selected navigation page.

    Args:
        model_inference: Loaded model inference object.

    Returns:
        Selected page label.
    """
    st.sidebar.markdown("## 🫁 Navigation")
    selected_page: str = st.sidebar.radio(
        "Go to",
        ["🏠 Overview", "📊 Performance", "🔍 Live Prediction", "🧠 Explainability"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    status_text: str = "✅ Loaded" if model_inference is not None else "❌ Not Loaded"
    device_name: str = _extract_device_name(model_inference) if model_inference is not None else "unknown"
    st.sidebar.info(f"**Model:** DenseNet121\n\n**Status:** {status_text}\n\n**Device:** {device_name}")

    st.sidebar.divider()
    st.sidebar.markdown("🤗 Hugging Face Space: [URL placeholder](https://huggingface.co/spaces/your-username/your-space)")
    st.sidebar.markdown("💻 GitHub Repository: [URL placeholder](https://github.com/your-username/your-repo)")
    st.sidebar.markdown("📧 Contact: [placeholder](mailto:you@example.com)")

    st.sidebar.divider()
    st.sidebar.caption("Built with PyTorch & Streamlit")
    return selected_page


def main() -> None:
    """Run Streamlit dashboard.

    Args:
        None.

    Returns:
        None.
    """
    _inject_custom_css()
    data_summary: dict[str, Any] | None = load_json_file(REPORTS_DIR / "data_summary.json")
    evaluation_report: dict[str, Any] | None = load_json_file(REPORTS_DIR / "evaluation_report.json")
    training_results: dict[str, Any] | None = load_json_file(REPORTS_DIR / "training_results.json")

    try:
        model_inference: Any = load_model_inference()
    except (FileNotFoundError, ImportError, RuntimeError, KeyError, OSError, ValueError, yaml.YAMLError) as exc:
        LOGGER.error("Model loading failed: %s\n%s", exc, traceback.format_exc())
        model_inference = None

    selected_page: str = render_sidebar(model_inference)
    if selected_page == "🏠 Overview":
        render_overview_page(data_summary)
    elif selected_page == "📊 Performance":
        render_performance_page(evaluation_report, training_results)
    elif selected_page == "🔍 Live Prediction":
        render_live_prediction_page(model_inference)
    else:
        render_explainability_page()


if __name__ == "__main__":
    main()
