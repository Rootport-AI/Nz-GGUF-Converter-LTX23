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

import hashlib
import logging
import sys
from pathlib import Path
from typing import Union

from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)

# Read size for the streaming SHA-256 pass. 8 MiB keeps the resident buffer
# negligible even for the ~43 GB source model.
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class RemoteSizeMismatchError(RuntimeError):
    """Raised when the remote file size does not match config.toml's expected_size.

    This usually means the repository's file has been replaced/updated upstream
    since ``expected_size`` was recorded, so the download is aborted rather than
    silently fetching a different file.
    """


class LocalSizeMismatchError(RuntimeError):
    """Raised when the downloaded file's own size does not match ``expected_size``."""


class Sha256MismatchError(RuntimeError):
    """Raised when a file's SHA-256 does not match the pinned digest.

    Pinning a digest is the last line of defence against silently converting a
    different revision of a model; see ``PRUNAVAED_WORKORDER.md`` §0-5 / §5.2.
    """


def sha256_of_file(path: Union[str, Path], chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """Return the lowercase hex SHA-256 of ``path``, read in bounded chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Union[str, Path], expected_sha256: str, *, label: str = "file") -> str:
    """Compute ``path``'s SHA-256 and raise unless it equals ``expected_sha256``.

    Deliberately a ``raise`` and not an ``assert``: ``assert`` statements are
    stripped by ``python -O``, which would silently disable the only check that
    can tell two revisions of the same model apart.
    """
    actual = sha256_of_file(path)
    if actual.lower() != expected_sha256.lower():
        raise Sha256MismatchError(
            f"SHA-256 mismatch for {label} {path}: got {actual}, expected "
            f"{expected_sha256}. The file is not the pinned revision -- refusing "
            "to continue. Delete it and download again, or update the pinned "
            "digest only after manually confirming the new file is correct."
        )
    return actual


def download(
    repo_id: str,
    filename: str,
    local_dir: Union[str, Path],
    expected_size: int,
    revision: Union[str, None] = None,
    sha256: Union[str, None] = None,
) -> Path:
    """Download ``filename`` from ``repo_id`` directly into ``local_dir``.

    Args:
        repo_id: Hugging Face Hub repo id, e.g. "SulphurAI/Sulphur-2-base".
        filename: File path within the repo, e.g. "sulphur_distil_bf16.safetensors".
        local_dir: Destination directory. The file is placed at
            ``local_dir/filename`` (created if missing).
        expected_size: Expected size of the file in bytes, from config.toml.
        revision: Optional git revision (commit hash / branch / tag) to pin the
            download to. ``None`` keeps the historical behaviour (repo default
            branch), so existing GGUF flows are unaffected.
        sha256: Optional pinned lowercase hex SHA-256 of the file's *content*.
            When given, it is verified both on the already-present fast path and
            after a fresh download.

    Returns:
        Path to the downloaded (or already-present) file.

    Raises:
        RemoteSizeMismatchError: if the size reported by the Hub does not match
            ``expected_size``.
        FileNotFoundError: if ``filename`` does not exist in ``repo_id``.
        LocalSizeMismatchError: if the downloaded file's actual size does not
            match ``expected_size``.
        Sha256MismatchError: if ``sha256`` was given and does not match.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    destination = local_dir / filename

    # 1. Already downloaded with the right size -> skip the transfer, but still
    #    prove the content is the pinned revision when a digest is configured.
    if destination.is_file() and destination.stat().st_size == expected_size:
        logger.info(
            "%s already present with expected size (%d bytes); skipping download.",
            destination,
            expected_size,
        )
        if sha256:
            logger.info("Verifying SHA-256 of the existing file (this reads it once)...")
            verify_sha256(destination, sha256, label="existing file")
            logger.info("SHA-256 OK: %s", sha256)
        return destination

    # 2. Verify remote size before downloading (repo may have changed upstream).
    api = HfApi()
    infos = api.get_paths_info(repo_id, [filename], revision=revision)
    match = next((info for info in infos if info.path == filename), None)
    if match is None:
        where = f"{repo_id!r}" if revision is None else f"{repo_id!r}@{revision}"
        raise FileNotFoundError(f"{filename!r} not found in repo {where} on the Hub.")

    remote_size = match.size
    if remote_size != expected_size:
        raise RemoteSizeMismatchError(
            f"Remote size mismatch for {repo_id}/{filename}: "
            f"config.toml expects {expected_size} bytes but the Hub reports "
            f"{remote_size} bytes. The repository file may have been replaced "
            "upstream -- aborting download. Update config.toml's expected_size "
            "only after manually confirming the new file is correct."
        )

    # 2b. Best effort: the Hub reports the LFS digest of the blob, so a pinned
    #     mismatch can be caught *before* spending the bandwidth.
    remote_lfs = getattr(match, "lfs", None)
    remote_sha = getattr(remote_lfs, "sha256", None) if remote_lfs is not None else None
    if sha256 and remote_sha and remote_sha.lower() != sha256.lower():
        raise Sha256MismatchError(
            f"Remote SHA-256 mismatch for {repo_id}/{filename}"
            f"{'' if revision is None else '@' + revision}: the Hub reports "
            f"{remote_sha} but the pinned digest is {sha256}. Aborting before "
            "downloading."
        )

    # 3. Download straight into local_dir (no global cache duplication -- see
    #    module docstring). huggingface_hub resumes automatically from any
    #    existing *.incomplete file on retry.
    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(local_dir), revision=revision
        )
    )

    # 4. Verify the downloaded file's actual size. This is a ``raise`` and not an
    #    ``assert`` on purpose -- ``python -O`` strips ``assert``, and this is the
    #    last check standing between a truncated download and a broken model.
    actual_size = downloaded_path.stat().st_size
    if actual_size != expected_size:
        raise LocalSizeMismatchError(
            f"Downloaded file size mismatch: {downloaded_path} is {actual_size} bytes, "
            f"expected {expected_size} bytes."
        )

    # 5. Verify the pinned content digest.
    if sha256:
        logger.info("Verifying SHA-256 of the downloaded file...")
        verify_sha256(downloaded_path, sha256, label="downloaded file")
        logger.info("SHA-256 OK: %s", sha256)

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
    revision = source.get("revision")
    sha256 = source.get("sha256")

    try:
        path = download(repo_id, filename, local_dir, expected_size, revision, sha256)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return 1

    logger.info("Download complete: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
