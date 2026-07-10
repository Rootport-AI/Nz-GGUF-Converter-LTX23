"""Download the source safetensors file from the Hugging Face Hub.

This module implements the "download" step of the GGUF conversion pipeline
(see ``config.toml`` / README.md, ``run.bat download``).

Key requirement: the source file is ~43GB, so it must be written directly
into the project's ``safetensors/`` directory and must NOT be duplicated
into the global Hugging Face cache (``~/.cache/huggingface`` or ``HF_HOME``).

``huggingface_hub`` (>=0.23) supports this natively: when ``hf_hub_download``
is called with ``local_dir=...``, it bypasses the global blob-store cache
entirely and downloads straight into ``local_dir``. Only small bookkeeping
metadata (etag/commit hash, a few bytes per file) is written under
``local_dir/.cache/huggingface/download/`` -- the actual file content never
touches the global cache. See ``huggingface_hub.file_download.hf_hub_download``
docstring: "If `local_dir` is provided, ... the `cache_dir` will not be used
and a `.cache/huggingface/` folder will be created at the root of `local_dir`
to store some metadata related to the downloaded files."
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union

from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)


class RemoteSizeMismatchError(RuntimeError):
    """Raised when the remote file size does not match config.toml's expected_size.

    This usually means the repository's file has been replaced/updated upstream
    since ``expected_size`` was recorded, so the download is aborted rather than
    silently fetching a different file.
    """


def download(
    repo_id: str,
    filename: str,
    local_dir: Union[str, Path],
    expected_size: int,
) -> Path:
    """Download ``filename`` from ``repo_id`` directly into ``local_dir``.

    Args:
        repo_id: Hugging Face Hub repo id, e.g. "SulphurAI/Sulphur-2-base".
        filename: File path within the repo, e.g. "sulphur_distil_bf16.safetensors".
        local_dir: Destination directory. The file is placed at
            ``local_dir/filename`` (created if missing).
        expected_size: Expected size of the file in bytes, from config.toml.

    Returns:
        Path to the downloaded (or already-present) file.

    Raises:
        RemoteSizeMismatchError: if the size reported by the Hub does not match
            ``expected_size``.
        FileNotFoundError: if ``filename`` does not exist in ``repo_id``.
        AssertionError: if the downloaded file's actual size does not match
            ``expected_size``.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    destination = local_dir / filename

    # 1. Already downloaded with the right size -> skip entirely.
    if destination.is_file() and destination.stat().st_size == expected_size:
        logger.info(
            "%s already present with expected size (%d bytes); skipping download.",
            destination,
            expected_size,
        )
        return destination

    # 2. Verify remote size before downloading (repo may have changed upstream).
    api = HfApi()
    infos = api.get_paths_info(repo_id, [filename])
    match = next((info for info in infos if info.path == filename), None)
    if match is None:
        raise FileNotFoundError(f"{filename!r} not found in repo {repo_id!r} on the Hub.")

    remote_size = match.size
    if remote_size != expected_size:
        raise RemoteSizeMismatchError(
            f"Remote size mismatch for {repo_id}/{filename}: "
            f"config.toml expects {expected_size} bytes but the Hub reports "
            f"{remote_size} bytes. The repository file may have been replaced "
            "upstream -- aborting download. Update config.toml's expected_size "
            "only after manually confirming the new file is correct."
        )

    # 3. Download straight into local_dir (no global cache duplication -- see
    #    module docstring). huggingface_hub resumes automatically from any
    #    existing *.incomplete file on retry.
    downloaded_path = Path(
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))
    )

    # 4. Verify the downloaded file's actual size.
    actual_size = downloaded_path.stat().st_size
    assert actual_size == expected_size, (
        f"Downloaded file size mismatch: {downloaded_path} is {actual_size} bytes, "
        f"expected {expected_size} bytes."
    )

    return downloaded_path


def _project_root() -> Path:
    """Return the project root directory (parent of the ``src`` package)."""
    return Path(__file__).resolve().parents[2]


def _load_config(config_path: Path) -> dict:
    try:
        import tomllib  # Python >= 3.11 (stdlib)
    except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
        import tomli as tomllib  # type: ignore[no-redef]

    with config_path.open("rb") as f:
        return tomllib.load(f)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = _project_root()
    config_path = root / "config.toml"
    config = _load_config(config_path)

    source = config["source"]
    repo_id = source["repo_id"]
    filename = source["filename"]
    expected_size = source["expected_size"]
    local_dir = root / source["local_dir"]

    try:
        path = download(repo_id, filename, local_dir, expected_size)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return 1

    logger.info("Download complete: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
