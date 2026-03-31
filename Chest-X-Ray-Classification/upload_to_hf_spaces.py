"""Upload the local Hugging Face Space bundle to the Hugging Face Hub."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Final

import httpx
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

REPO_ID: Final[str] = "chathurab1120/chest-xray-classifier"
REPO_TYPE: Final[str] = "space"
SPACE_SDK: Final[str] = "gradio"
SPACE_ROOT: Final[Path] = Path(__file__).resolve().parent / "huggingface_space"
MODEL_LOCAL: Final[Path] = SPACE_ROOT / "models" / "best_model.pth"
MODEL_REMOTE: Final[str] = "models/best_model.pth"
SPACE_URL: Final[str] = (
    "https://huggingface.co/spaces/chathurab1120/chest-xray-classifier"
)

LOGGER: logging.Logger = logging.getLogger("upload_to_hf_spaces")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.propagate = False

_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

MAX_UPLOAD_RETRIES: Final[int] = 3
RETRY_DELAY_SECONDS: Final[int] = 30
_TRANSIENT_HTTP_STATUSES: Final[frozenset[int]] = frozenset({502, 503, 504})


def _is_transient_upload_error(exc: BaseException) -> bool:
    """Return True if the error may resolve after a short wait (retry candidate).

    Args:
        exc: Exception raised by ``upload_file`` or the HTTP stack.

    Returns:
        True when the failure is a transient Hub or network condition.
    """
    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError)):
        return True
    if isinstance(exc, HfHubHTTPError):
        response = exc.response
        if response is not None and response.status_code in _TRANSIENT_HTTP_STATUSES:
            return True
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_HTTP_STATUSES
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
        ),
    ):
        return True
    return False


def upload_with_retry(
    api: HfApi,
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
    token: str,
) -> None:
    """Call ``upload_file`` with retries on transient Hub or network errors.

    On each transient failure, logs ``Retry N/MAX`` and waits ``RETRY_DELAY_SECONDS``
    before trying again, up to ``MAX_UPLOAD_RETRIES`` retries.

    Args:
        api: Authenticated ``HfApi`` instance.
        repo_id: Target Space repository id.
        local_path: Absolute path to the file on disk.
        path_in_repo: Destination path inside the Space repository.
        token: Hugging Face API token for Hub requests.

    Returns:
        None.

    Raises:
        BaseException: The last error if all retries are exhausted or the error is not
            transient (re-raised unchanged).
    """
    for attempt_index in range(MAX_UPLOAD_RETRIES + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                token=token,
            )
            return
        except Exception as exc:
            if not _is_transient_upload_error(exc):
                raise
            if attempt_index >= MAX_UPLOAD_RETRIES:
                raise exc
            LOGGER.warning(
                "Retry %d/%d for file %s - waiting %ds...",
                attempt_index + 1,
                MAX_UPLOAD_RETRIES,
                path_in_repo,
                RETRY_DELAY_SECONDS,
            )
            time.sleep(RETRY_DELAY_SECONDS)


def _require_hf_token() -> str:
    """Return ``HF_TOKEN`` from the environment (after ``load_dotenv()``).

    Args:
        None.

    Returns:
        Non-empty Hugging Face API token string.

    Raises:
        SystemExit: If ``HF_TOKEN`` is missing or empty (exit code 1).
    """
    token: str | None = os.getenv("HF_TOKEN")
    if token is not None:
        token = token.strip()
    if token:
        return token
    LOGGER.error("HF_TOKEN not found in .env file")
    raise SystemExit(1)


def _ensure_space_repo(api: HfApi, repo_id: str, token: str) -> None:
    """Create the Space repository on the Hub if it does not exist.

    Args:
        api: Authenticated ``HfApi`` instance.
        repo_id: Target repository id (``namespace/name``).
        token: Hugging Face API token for Hub requests.

    Returns:
        None.

    Raises:
        Exception: If repository creation fails for reasons other than "already exists".
    """
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            space_sdk=SPACE_SDK,
            exist_ok=True,
            token=token,
        )
        LOGGER.info("Space repo ready: %s", repo_id)
    except Exception as exc:
        LOGGER.exception("Failed to create or verify Space repo %s: %s", repo_id, exc)
        raise


def _collect_non_model_files(space_root: Path) -> list[tuple[Path, str]]:
    """Build (local path, repo-relative path) pairs for all files except the checkpoint.

    Args:
        space_root: Root directory of the local Space (``huggingface_space/``).

    Returns:
        List of upload pairs for app code, config, src scripts, and example images.

    Raises:
        FileNotFoundError: If ``space_root`` or a required file is missing.
    """
    if not space_root.is_dir():
        raise FileNotFoundError(f"Space directory not found: {space_root}")

    required_files: list[tuple[str, str]] = [
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
        ("README.md", "README.md"),
        ("configs/config.yaml", "configs/config.yaml"),
        ("src/03_model_training.py", "src/03_model_training.py"),
        ("src/04_evaluation.py", "src/04_evaluation.py"),
    ]
    pairs: list[tuple[Path, str]] = []
    for rel, remote in required_files:
        local_path: Path = space_root / rel
        if not local_path.is_file():
            raise FileNotFoundError(f"Required Space file missing: {local_path}")
        pairs.append((local_path, remote))

    for class_name in ("NORMAL", "PNEUMONIA"):
        class_dir: Path = space_root / "test_examples" / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Example directory missing: {class_dir}")
        image_paths: list[Path] = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ]
        if class_name == "NORMAL" and len(image_paths) < 1:
            raise FileNotFoundError(f"Need at least 1 image in {class_dir}")
        if class_name == "PNEUMONIA" and len(image_paths) < 2:
            raise FileNotFoundError(f"Need at least 2 images in {class_dir}")
        for image_path in image_paths:
            remote_path: str = (
                f"test_examples/{class_name}/{image_path.name}"
            )
            pairs.append((image_path, remote_path))

    return pairs


def _upload_single_file(
    api: HfApi,
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
    token: str,
) -> None:
    """Upload one file to the Space, logging structured errors on failure.

    Args:
        api: Authenticated ``HfApi`` instance.
        repo_id: Target Space repository id.
        local_path: Absolute path to the file on disk.
        path_in_repo: Destination path inside the Space repository.
        token: Hugging Face API token for Hub requests.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the local file is missing.
        PermissionError: If the local file cannot be read due to permissions.
        RuntimeError: If the Hub rejects the upload after logging.
    """
    try:
        upload_with_retry(api, repo_id, local_path, path_in_repo, token)
        LOGGER.info("Uploaded %s -> %s", local_path, path_in_repo)
    except FileNotFoundError as exc:
        LOGGER.error(
            "Failed to read local file (missing): %s | detail=%s",
            local_path,
            exc,
        )
        raise
    except PermissionError as exc:
        LOGGER.error(
            "Failed to read local file (permission denied): %s | detail=%s",
            local_path,
            exc,
        )
        raise
    except HfHubHTTPError as exc:
        response = exc.response
        status_code: int | None = (
            response.status_code if response is not None else None
        )
        url_str: str = str(response.url) if response is not None else "unknown"
        if status_code in _TRANSIENT_HTTP_STATUSES:
            LOGGER.error(
                "Upload failed after %d retries (HTTP): status=%s url=%s "
                "path_in_repo=%s detail=%s",
                MAX_UPLOAD_RETRIES,
                status_code,
                url_str,
                path_in_repo,
                exc,
            )
        else:
            LOGGER.error(
                "Hub HTTP error (no retry): status=%s url=%s path_in_repo=%s detail=%s",
                status_code,
                url_str,
                path_in_repo,
                exc,
            )
        raise RuntimeError(
            f"Upload failed for {local_path} ({path_in_repo})"
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        url_str = str(exc.request.url)
        if status_code in _TRANSIENT_HTTP_STATUSES:
            LOGGER.error(
                "Upload failed after %d retries (HTTP): status=%s url=%s "
                "path_in_repo=%s detail=%s",
                MAX_UPLOAD_RETRIES,
                status_code,
                url_str,
                path_in_repo,
                exc,
            )
        else:
            LOGGER.error(
                "HTTP error (no retry): status=%s url=%s path_in_repo=%s detail=%s",
                status_code,
                url_str,
                path_in_repo,
                exc,
            )
        raise RuntimeError(
            f"Upload failed for {local_path} ({path_in_repo})"
        ) from exc
    except (
        TimeoutError,
        ConnectionError,
        BrokenPipeError,
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
    ) as exc:
        LOGGER.error(
            "Upload failed after %d retries (network/timeout): path_in_repo=%s "
            "detail=%s",
            MAX_UPLOAD_RETRIES,
            path_in_repo,
            exc,
        )
        raise RuntimeError(
            f"Upload failed for {local_path} ({path_in_repo})"
        ) from exc


def _upload_model_weights(
    api: HfApi,
    repo_id: str,
    model_path: Path,
    token: str,
) -> None:
    """Log model size and upload ``best_model.pth`` to ``models/best_model.pth``.

    Args:
        api: Authenticated ``HfApi`` instance.
        repo_id: Target Space repository id.
        model_path: Local path to ``best_model.pth``.
        token: Hugging Face API token for Hub requests.

    Returns:
        None.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        RuntimeError: If upload fails.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    size_bytes: int = model_path.stat().st_size
    size_mb: float = size_bytes / (1024 * 1024)
    LOGGER.info(
        "Model checkpoint size: %.2f MB (%d bytes) — uploading to %s",
        size_mb,
        size_bytes,
        MODEL_REMOTE,
    )
    _upload_single_file(api, repo_id, model_path, MODEL_REMOTE, token)


def deploy_space_bundle() -> None:
    """Create the Space (if needed), upload app files, then upload model weights.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If authentication is missing (code 1).
        FileNotFoundError: If required local artifacts are missing.
        RuntimeError: If any upload step fails.
    """
    token: str = _require_hf_token()
    api: HfApi = HfApi(token=token)
    _ensure_space_repo(api, REPO_ID, token)

    try:
        file_pairs: list[tuple[Path, str]] = _collect_non_model_files(SPACE_ROOT)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        raise

    for local_path, remote in file_pairs:
        _upload_single_file(api, REPO_ID, local_path, remote, token)

    _upload_model_weights(api, REPO_ID, MODEL_LOCAL, token)

    LOGGER.info(
        "\n"
        "================================================\n"
        "DEPLOYMENT COMPLETE!\n"
        "================================================\n"
        "Space URL: %s\n"
        "The space will take 2-3 minutes to build.\n"
        "Check build logs at the URL above.\n"
        "================================================",
        SPACE_URL,
    )


def main() -> None:
    """CLI entry point for uploading the Space bundle.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: On authentication failure or unrecoverable errors.
    """
    try:
        deploy_space_bundle()
    except SystemExit:
        raise
    except Exception as exc:
        LOGGER.exception("Deployment failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
