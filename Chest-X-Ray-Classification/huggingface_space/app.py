"""Gradio web application for chest X-ray pneumonia classification (Hugging Face Spaces)."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import ChestXRayModel
from src.gradcam import GradCAM

LOGGER: logging.Logger = logging.getLogger("gradio_app")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    console_handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)
    LOGGER.propagate = False

SPACE_ROOT: Path = Path(__file__).resolve().parent
MODEL_RELATIVE: Path = Path("models") / "best_model.pth"
CONFIG_RELATIVE: Path = Path("configs") / "config.yaml"
TEST_EXAMPLES_RELATIVE: Path = Path("test_examples")


def download_model_if_needed(space_root: Path) -> None:
    """Verify model weights exist; exit gracefully if missing (e.g. before upload).

    Args:
        space_root: Directory containing ``app.py`` (Hugging Face Space root).

    Returns:
        None.

    Raises:
        SystemExit: With code 1 if the model file is absent.
    """
    model_path: Path = space_root / MODEL_RELATIVE
    if model_path.exists():
        return
    message: str = "Model not found. Please upload model weights."
    LOGGER.error("%s Expected path: %s", message, model_path)
    sys.exit(1)


download_model_if_needed(SPACE_ROOT)


class ModelInference:
    """Inference wrapper for chest X-ray model and Grad-CAM.

    Args:
        model_path: Path to best model checkpoint.
        config_path: Path to project config file.

    Raises:
        FileNotFoundError: If required model/config files are missing.
        KeyError: If config fields are missing.
        RuntimeError: If model loading fails.
    """

    def __init__(self, model_path: Path, config_path: Path) -> None:
        self.model_path: Path = model_path
        self.config_path: Path = config_path
        self.config: dict[str, Any] = self._load_config(config_path)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = self._load_model(model_path)
        self.transform: transforms.Compose = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.4693, 0.4693, 0.4693], [0.2270, 0.2270, 0.2270]),
            ]
        )

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        """Load YAML configuration file.

        Args:
            config_path: Path to YAML configuration.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If config content is invalid.
            yaml.YAMLError: If YAML parse fails.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        try:
            with config_path.open("r", encoding="utf-8") as file_pointer:
                loaded_config: Any = yaml.safe_load(file_pointer)
        except yaml.YAMLError:
            raise
        except OSError as exc:
            raise ValueError(f"Could not read config file: {config_path}") from exc
        if not isinstance(loaded_config, dict):
            raise ValueError("Config content must be a dictionary.")
        return loaded_config

    def _load_model(self, model_path: Path) -> nn.Module:
        """Load trained model checkpoint into DenseNet architecture.

        Args:
            model_path: Path to best checkpoint.

        Returns:
            Loaded model in eval mode.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
            KeyError: If checkpoint state dict key is missing.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        model: nn.Module = ChestXRayModel(
            model_name=str(self.config["model"]["name"]),
            num_classes=int(self.config["model"]["num_classes"]),
            pretrained=bool(self.config["model"]["pretrained"]),
        )
        checkpoint_data: dict[str, Any] = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.to(self.device)
        model.eval()
        LOGGER.info("Loaded model from %s on device %s", model_path, self.device)
        return model

    def predict(self, image: Image.Image) -> dict[str, float | str]:
        """Run single-image prediction.

        Args:
            image: Input chest X-ray image.

        Returns:
            Dictionary containing class prediction and probabilities.

        Raises:
            ValueError: If image is invalid.
            RuntimeError: If inference fails.
        """
        if image is None:
            raise ValueError("Input image is required.")
        rgb_image: Image.Image = image.convert("RGB")
        image_tensor: Tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits: Tensor = self.model(image_tensor)
            probabilities: Tensor = torch.softmax(logits, dim=1)

        pneumonia_prob: float = float(probabilities[0, 1].item())
        normal_prob: float = float(probabilities[0, 0].item())
        predicted_index: int = int(torch.argmax(probabilities, dim=1).item())
        predicted_class: str = "PNEUMONIA" if predicted_index == 1 else "NORMAL"
        confidence: float = float(probabilities[0, predicted_index].item())
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "normal_prob": normal_prob,
            "pneumonia_prob": pneumonia_prob,
        }

    def generate_gradcam(self, image: Image.Image) -> Image.Image:
        """Generate side-by-side original and Grad-CAM overlay image.

        Args:
            image: Input chest X-ray image.

        Returns:
            Side-by-side PIL image (original and Grad-CAM overlay).

        Raises:
            ValueError: If input image is missing.
            RuntimeError: If Grad-CAM generation fails.
        """
        if image is None:
            raise ValueError("Input image is required for Grad-CAM.")
        rgb_image: Image.Image = image.convert("RGB").resize((224, 224))
        image_tensor: Tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        gradcam_generator: Any = GradCAM(self.model, target_layer_name="features.denseblock4")
        try:
            heatmap: np.ndarray = gradcam_generator.compute(image_tensor)
        finally:
            gradcam_generator.remove_hooks()

        original_array: np.ndarray = np.asarray(rgb_image, dtype=np.uint8)
        heatmap_uint8: np.ndarray = np.uint8(255 * heatmap)
        heatmap_color: np.ndarray = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb: np.ndarray = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay: np.ndarray = cv2.addWeighted(original_array, 0.5, heatmap_rgb, 0.5, 0)

        combined: Image.Image = Image.new("RGB", (224 * 2, 224 + 30), color=(255, 255, 255))
        combined.paste(Image.fromarray(original_array), (0, 30))
        combined.paste(Image.fromarray(overlay), (224, 30))

        draw: ImageDraw.ImageDraw = ImageDraw.Draw(combined)
        draw.text((70, 8), "Original", fill=(0, 0, 0))
        draw.text((258, 8), "AI Focus Area", fill=(0, 0, 0))
        return combined


def _format_assessment(pneumonia_prob: float, normal_prob: float) -> str:
    """Create risk assessment text from class probabilities.

    Args:
        pneumonia_prob: Predicted pneumonia probability.
        normal_prob: Predicted normal probability.

    Returns:
        Human-readable risk assessment text.
    """
    if pneumonia_prob > 0.70:
        assessment: str = (
            "High likelihood of pneumonia detected.\n"
            "Please consult a radiologist immediately."
        )
    elif 0.40 <= pneumonia_prob <= 0.70:
        assessment = (
            "Moderate indicators present.\n"
            "Further clinical evaluation recommended."
        )
    elif normal_prob > 0.70:
        assessment = (
            "Low indicators of pneumonia.\n"
            "Routine monitoring advised."
        )
    else:
        assessment = (
            "Uncertain prediction confidence.\n"
            "Further clinical evaluation recommended."
        )
    return (
        f"{assessment}\n\n"
        "⚠️ Research use only.\n"
        "Not a substitute for professional medical diagnosis."
    )


def _find_example_images() -> list[list[str]]:
    """Build Gradio example rows using absolute paths under the Space root.

    Relative paths break on Hugging Face Spaces because Gradio copies examples
    into ``/tmp/gradio/...``; resolved absolute paths stay valid.

    Args:
        None.

    Returns:
        Rows for ``gr.Examples``: each row is ``[absolute_path]``.
    """
    examples: list[list[str]] = []
    normal_dir: Path = SPACE_ROOT / TEST_EXAMPLES_RELATIVE / "NORMAL"
    pneumonia_dir: Path = SPACE_ROOT / TEST_EXAMPLES_RELATIVE / "PNEUMONIA"

    normal_images: list[Path] = (
        sorted(normal_dir.glob("*.jpeg"))
        + sorted(normal_dir.glob("*.jpg"))
        + sorted(normal_dir.glob("*.png"))
    )
    if normal_images:
        examples.append([str(normal_images[0].resolve())])

    pneumonia_images: list[Path] = (
        sorted(pneumonia_dir.glob("*.jpeg"))
        + sorted(pneumonia_dir.glob("*.jpg"))
        + sorted(pneumonia_dir.glob("*.png"))
    )
    for img in pneumonia_images[:2]:
        examples.append([str(img.resolve())])

    return examples


MODEL_INFERENCE: ModelInference = ModelInference(
    model_path=SPACE_ROOT / MODEL_RELATIVE,
    config_path=SPACE_ROOT / CONFIG_RELATIVE,
)


def predict_image(image: Image.Image | None) -> tuple[str, str, Image.Image | None, str]:
    """Gradio prediction handler.

    Args:
        image: Uploaded PIL image.

    Returns:
        Tuple containing diagnosis, score text, Grad-CAM image, and assessment text.
    """
    if image is None:
        return ("Please upload a chest X-ray image", "", None, "")
    try:
        prediction: dict[str, float | str] = MODEL_INFERENCE.predict(image)
        predicted_class: str = str(prediction["predicted_class"])
        normal_prob: float = float(prediction["normal_prob"])
        pneumonia_prob: float = float(prediction["pneumonia_prob"])

        diagnosis_text: str = "🔴 PNEUMONIA DETECTED" if predicted_class == "PNEUMONIA" else "✅ NORMAL"
        score_text: str = f"PNEUMONIA: {pneumonia_prob*100:.1f}%  |  NORMAL: {normal_prob*100:.1f}%"
        assessment_text: str = _format_assessment(pneumonia_prob, normal_prob)
        gradcam_image: Image.Image = MODEL_INFERENCE.generate_gradcam(image)
        return diagnosis_text, score_text, gradcam_image, assessment_text
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        LOGGER.error("Inference failure: %s\n%s", exc, traceback.format_exc())
        return (
            "Unable to analyze image at this time.",
            "Please try another image or retry in a moment.",
            None,
            "⚠️ Research use only.\nNot a substitute for professional medical diagnosis.",
        )


def build_interface() -> gr.Blocks:
    """Build the Gradio Blocks interface.

    Args:
        None.

    Returns:
        Configured Gradio Blocks app.
    """
    with gr.Blocks(title="Chest X-Ray Pneumonia Classifier") as demo:
        gr.Markdown("# 🫁 Chest X-Ray Pneumonia Classifier")
        gr.Markdown(
            "DenseNet121 transfer learning model with Grad-CAM explainability. "
            "Test performance: Accuracy 87.8%, Precision 84.0%, Recall 99.5%, "
            "F1 91.1%, AUC-ROC 96.8%, Specificity 68.4%."
        )
        gr.Markdown(
            "### 🚨 **Research Use Only**\n"
            "This AI tool is for educational and research demonstration only, not clinical diagnosis."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input: gr.Image = gr.Image(type="pil", label="Upload Chest X-Ray Image")
                analyze_button: gr.Button = gr.Button("🔍 Analyze X-Ray", variant="primary")
                example_images: list[list[str]] = _find_example_images()
                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=[image_input],
                        label="Example Test Images",
                    )

            with gr.Column(scale=1):
                diagnosis_box: gr.Textbox = gr.Textbox(label="📋 Diagnosis", interactive=False)
                scores_box: gr.Textbox = gr.Textbox(label="📊 Confidence Scores", interactive=False)
                assessment_box: gr.Textbox = gr.Textbox(
                    label="⚕️ Risk Assessment", interactive=False, lines=4
                )
                gradcam_box: gr.Image = gr.Image(label="🧠 Grad-CAM — Where the AI is looking")

        analyze_button.click(
            fn=predict_image,
            inputs=image_input,
            outputs=[diagnosis_box, scores_box, gradcam_box, assessment_box],
        )

        with gr.Accordion("ℹ️ How does this AI work?", open=False):
            gr.Markdown(
                "- The model uses **DenseNet121** with transfer learning to identify patterns linked to pneumonia.\n"
                "- Transfer learning starts from prior visual knowledge and adapts it to chest X-ray features.\n"
                "- **Grad-CAM** highlights where the network focuses; red areas indicate higher model attention.\n"
                "- In medical screening, high **recall (99.5%)** is critical to minimize missed pneumonia cases.\n\n"
                "**Model Stats:** Accuracy 87.8%, Precision 84.0%, Recall 99.5%, F1 91.1%, "
                "AUC-ROC 96.8%, Specificity 68.4%."
            )

        gr.Markdown(
            "Dataset credit: Kaggle Chest X-Ray Images (Pneumonia)\n\n"
            "View full project on GitHub: "
            "[Data-Science-Projects-New]"
            "(https://github.com/chathurab1120/Data-Science-Projects-New/tree/main/Chest-X-Ray-Classification)"
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(ssr_mode=False)
else:
    # Hugging Face Spaces may import this module; expose ``demo`` without launching here.
    demo = build_interface()
