"""Copy trained artifacts into ``huggingface_space/`` and log deployment steps."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

LOGGER: logging.Logger = logging.getLogger("deploy_to_hf")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    _handler: logging.StreamHandler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.propagate = False


def _project_root() -> Path:
    """Return the Chest X-Ray project root (directory containing this script).

    Args:
        None.

    Returns:
        Absolute path to project root.
    """
    return Path(__file__).resolve().parent


def _copy_example_images(
    source_test_dir: Path,
    dest_examples_dir: Path,
) -> tuple[Path | None, list[Path]]:
    """Copy one NORMAL and two PNEUMONIA images into the HF ``test_examples`` tree.

    Args:
        source_test_dir: Path to ``.../chest_xray/test`` (contains ``NORMAL`` / ``PNEUMONIA``).
        dest_examples_dir: ``huggingface_space/test_examples`` root.

    Returns:
        Tuple of (single normal path or None, list of up to two pneumonia paths).

    Raises:
        FileNotFoundError: If ``source_test_dir`` does not exist.
    """
    if not source_test_dir.is_dir():
        raise FileNotFoundError(f"Test split not found: {source_test_dir}")
    normal_src: Path = source_test_dir / "NORMAL"
    pneumonia_src: Path = source_test_dir / "PNEUMONIA"
    dest_normal: Path = dest_examples_dir / "NORMAL"
    dest_pneumonia: Path = dest_examples_dir / "PNEUMONIA"
    dest_normal.mkdir(parents=True, exist_ok=True)
    dest_pneumonia.mkdir(parents=True, exist_ok=True)

    normal_files: list[Path] = sorted(normal_src.glob("*.jp*g"))
    pneumonia_files: list[Path] = sorted(pneumonia_src.glob("*.jp*g"))

    copied_normal: Path | None = None
    if normal_files:
        copied_normal = dest_normal / normal_files[0].name
        shutil.copy2(normal_files[0], copied_normal)
        LOGGER.info("Copied example NORMAL image to %s", copied_normal)

    copied_pneumonia: list[Path] = []
    for src in pneumonia_files[:2]:
        dest_path: Path = dest_pneumonia / src.name
        shutil.copy2(src, dest_path)
        copied_pneumonia.append(dest_path)
        LOGGER.info("Copied example PNEUMONIA image to %s", dest_path)

    return copied_normal, copied_pneumonia


def copy_artifacts_to_huggingface_space(project_root: Path) -> None:
    """Verify checkpoint exists, copy model and example images into ``huggingface_space/``.

    Args:
        project_root: Chest X-Ray project root directory.

    Returns:
        None.

    Raises:
        FileNotFoundError: If ``models/checkpoints/best_model.pth`` is missing.
        OSError: If copy operations fail.
    """
    checkpoint: Path = project_root / "models" / "checkpoints" / "best_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint}. Run training before deploying."
        )

    hf_root: Path = project_root / "huggingface_space"
    hf_models: Path = hf_root / "models"
    hf_models.mkdir(parents=True, exist_ok=True)
    dest_model: Path = hf_models / "best_model.pth"
    shutil.copy2(checkpoint, dest_model)
    LOGGER.info("Copied model to %s", dest_model)

    data_dir: Path = project_root / "data" / "raw" / "chest_xray" / "test"
    examples_dir: Path = hf_root / "test_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    try:
        _copy_example_images(data_dir, examples_dir)
    except FileNotFoundError as exc:
        LOGGER.warning("Could not copy example images: %s", exc)


def log_hf_deployment_instructions() -> None:
    """Log manual Hugging Face Spaces deployment steps to the logger.

    Args:
        None.

    Returns:
        None.
    """
    instructions: str = """
========================================
HUGGING FACE DEPLOYMENT STEPS:
========================================
1. Install HF CLI: pip install huggingface_hub
2. Login: huggingface-cli login
3. Create a new Space at: https://huggingface.co/new-space
   - Space name: chest-xray-classifier
   - SDK: Gradio
   - License: MIT
4. Push files:
   cd huggingface_space
   git init
   git add .
   git commit -m "Initial deployment"
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/chest-xray-classifier
   git push origin main
5. Upload model weights separately (large file):
   python -c "
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_file(
       path_or_fileobj='models/checkpoints/best_model.pth',
       path_in_repo='models/best_model.pth',
       repo_id='YOUR_USERNAME/chest-xray-classifier',
       repo_type='space'
   )
   "
========================================
"""
    LOGGER.info(instructions.strip())


def main() -> None:
    """Entry point: copy artifacts and print deployment instructions.

    Args:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the trained checkpoint is missing.
        OSError: If file operations fail.
    """
    root: Path = _project_root()
    try:
        copy_artifacts_to_huggingface_space(root)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        raise
    log_hf_deployment_instructions()


if __name__ == "__main__":
    main()
