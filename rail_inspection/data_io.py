from __future__ import annotations

import logging
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from rail_inspection.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def download_file(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    if destination.exists():
        LOGGER.info("Using cached file: %s", destination)
        return destination

    LOGGER.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as out_file, tqdm(
            total=total, unit="B", unit_scale=True, desc=destination.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                out_file.write(chunk)
                bar.update(len(chunk))
    return destination


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    suffix = "".join(archive_path.suffixes).lower()

    if suffix.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    elif suffix.endswith(".tar.gz") or suffix.endswith(".tgz") or suffix.endswith(".tar"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    LOGGER.info("Extracted %s to %s", archive_path, output_dir)
    return output_dir

