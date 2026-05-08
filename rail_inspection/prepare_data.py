from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rail_inspection.config import DatasetConfig
from rail_inspection.data_io import download_file, extract_archive
from rail_inspection.utils import ensure_dir

LOGGER = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FALLBACK_RAIL_URLS = [
    "https://huggingface.co/datasets/sleepysaurus/RailRakshak-Track-Detection-Data/resolve/main/Rail-DB.zip",
    "https://datasetdownloadslinks.s3.us-east-1.amazonaws.com/Download%20Datasets/"
    "Railway%20Track%20fault%20Detection%20Resized%20%28224%20X%20224%29.zip",
]


def _download_rail_with_fallback(cfg: DatasetConfig) -> Path:
    urls: list[str] = []
    if cfg.rail_url:
        urls.append(cfg.rail_url)
    urls.extend([u for u in FALLBACK_RAIL_URLS if u not in urls])

    last_error: Exception | None = None
    for idx, url in enumerate(urls):
        archive_path = cfg.raw_root / f"rail_dataset_{idx}.zip"
        try:
            archive = download_file(url, archive_path)
            return extract_archive(archive, cfg.raw_root / "rail")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.warning("Rail dataset download failed for URL %s: %s", url, exc)
            continue

    raise RuntimeError(f"All rail dataset URLs failed. Last error: {last_error}")


def _find_image_label_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

    for image_path in images:
        candidates = [
            image_path.with_suffix(".txt"),
            Path(str(image_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt"),
            image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        ]
        label_path = next((c for c in candidates if c.exists()), None)
        if label_path:
            pairs.append((image_path, label_path))
    return pairs


def _find_classification_images(root: Path) -> tuple[list[Path], dict[str, int]]:
    """Find class-folder style images and map folder names to class IDs."""
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    class_to_id: dict[str, int] = {}
    filtered: list[Path] = []
    for img in images:
        class_name = img.parent.name.strip().lower().replace(" ", "_")
        if not class_name:
            continue
        if class_name not in class_to_id:
            class_to_id[class_name] = len(class_to_id)
        filtered.append(img)
    return filtered, class_to_id


def _parse_label_classes(label_path: Path) -> list[int]:
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            classes.append(int(float(parts[0])))
    return classes


def _write_yolo_label(path: Path, class_id: int, x: float, y: float, w: float, h: float) -> None:
    path.write_text(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def _generate_spectrogram(wav_path: Path, out_path: Path) -> None:
    signal, sr = librosa.load(str(wav_path), sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mel_db, aspect="auto", origin="lower")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _collect_rail_data(cfg: DatasetConfig) -> tuple[list[tuple[Path, Path]], list[str]]:
    rail_dir = cfg.rail_local_dir
    if not rail_dir.exists() and cfg.rail_url:
        rail_dir = _download_rail_with_fallback(cfg)
    elif not rail_dir.exists():
        raise FileNotFoundError(
            f"Rail dataset not found at {cfg.rail_local_dir}. Set --rail-url or place dataset locally."
        )

    pairs = _find_image_label_pairs(rail_dir)
    if pairs:
        max_class = -1
        for _, label in pairs:
            classes = _parse_label_classes(label)
            if classes:
                max_class = max(max_class, max(classes))
        rail_names = [f"rail_class_{i}" for i in range(max_class + 1 if max_class >= 0 else 1)]
        LOGGER.info("Rail data (YOLO): %s pairs, %s classes", len(pairs), len(rail_names))
        return pairs, rail_names

    # Fallback: class-folder dataset (classification-like). We convert to YOLO detection labels.
    cls_images, class_map = _find_classification_images(rail_dir)
    if not cls_images and cfg.rail_url and rail_dir == cfg.rail_local_dir:
        LOGGER.warning("No valid rail images in local dir. Trying automatic rail dataset download...")
        rail_dir = _download_rail_with_fallback(cfg)
        pairs = _find_image_label_pairs(rail_dir)
        if pairs:
            max_class = -1
            for _, label in pairs:
                classes = _parse_label_classes(label)
                if classes:
                    max_class = max(max_class, max(classes))
            rail_names = [f"rail_class_{i}" for i in range(max_class + 1 if max_class >= 0 else 1)]
            LOGGER.info("Rail data (YOLO): %s pairs, %s classes", len(pairs), len(rail_names))
            return pairs, rail_names
        cls_images, class_map = _find_classification_images(rail_dir)
    if not cls_images:
        raise RuntimeError(f"No rail images found in {rail_dir}")

    converted_labels_root = ensure_dir(cfg.processed_root / "tmp_rail_labels")
    converted_pairs: list[tuple[Path, Path]] = []
    for image_path in cls_images:
        class_name = image_path.parent.name.strip().lower().replace(" ", "_")
        class_id = class_map[class_name]
        unique_id = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:10]
        label_path = converted_labels_root / f"{image_path.stem}_{unique_id}.txt"
        _write_yolo_label(label_path, class_id, 0.5, 0.5, 1.0, 1.0)
        converted_pairs.append((image_path, label_path))

    rail_names = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    LOGGER.info(
        "Rail data (class-folder converted): %s images, %s classes",
        len(converted_pairs),
        len(rail_names),
    )
    return converted_pairs, rail_names


def _collect_audio_data(cfg: DatasetConfig, rail_class_count: int) -> tuple[list[dict], list[str]]:
    audio_archive = download_file(cfg.audio_url, cfg.raw_root / "UrbanSound8K.tar.gz")
    extracted_audio_root = cfg.raw_root / "audio"
    audio_root = extracted_audio_root if extracted_audio_root.exists() else extract_archive(audio_archive, extracted_audio_root)

    meta_csv = next(audio_root.rglob("UrbanSound8K.csv"), None)
    if not meta_csv:
        raise RuntimeError("UrbanSound8K metadata CSV not found after extraction.")

    entries: list[dict] = []
    class_map: dict[str, int] = {}

    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["class"].strip()
            if class_name not in class_map:
                class_map[class_name] = rail_class_count + len(class_map)

            fold = int(row["fold"])
            split = "train" if fold <= 8 else ("val" if fold == 9 else "test")
            wav_candidates = [p for p in audio_root.rglob(row["slice_file_name"]) if f"fold{fold}" in str(p.parent)]
            wav = wav_candidates[0] if wav_candidates else None
            if wav is None:
                continue
            entries.append(
                {
                    "wav_path": wav,
                    "split": split,
                    "class_id": class_map[class_name],
                    "class_name": class_name,
                }
            )

    sorted_audio_names = [k for k, _ in sorted(class_map.items(), key=lambda x: x[1])]
    LOGGER.info("Audio data: %s files, %s classes", len(entries), len(sorted_audio_names))
    return entries, sorted_audio_names


def _copy_pair_to_split(
    image_path: Path,
    label_path: Path,
    split: str,
    images_root: Path,
    labels_root: Path,
    prefix: str = "",
) -> None:
    dst_img = images_root / split / f"{prefix}{image_path.stem}{image_path.suffix.lower()}"
    dst_lbl = labels_root / split / f"{prefix}{image_path.stem}.txt"
    ensure_dir(dst_img.parent)
    ensure_dir(dst_lbl.parent)
    shutil.copy2(image_path, dst_img)
    shutil.copy2(label_path, dst_lbl)


def build_multimodal_yolo_dataset(
    cfg: DatasetConfig,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    if abs((train_ratio + val_ratio) - 0.9) > 1e-8:
        raise ValueError("Expected test ratio to be 0.1. Set train+val = 0.9")

    output_root = ensure_dir(cfg.processed_root / "yolo_multimodal")
    images_root = ensure_dir(output_root / "images")
    labels_root = ensure_dir(output_root / "labels")

    rail_pairs, rail_names = _collect_rail_data(cfg)
    audio_entries, audio_names = _collect_audio_data(cfg, rail_class_count=len(rail_names))

    split_buckets: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    rail_indices = list(range(len(rail_pairs)))
    y = []
    for _, lbl in rail_pairs:
        classes = _parse_label_classes(lbl)
        y.append(classes[0] if classes else 0)

    try:
        train_idx, temp_idx = train_test_split(
            rail_indices, train_size=train_ratio, random_state=seed, stratify=y
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            rail_indices, train_size=train_ratio, random_state=seed, stratify=None
        )
    temp_labels = [y[idx] for idx in temp_idx]
    rel_val_ratio = val_ratio / (1.0 - train_ratio)
    try:
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=rel_val_ratio, random_state=seed, stratify=temp_labels
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=rel_val_ratio, random_state=seed, stratify=None
        )

    for idx in train_idx:
        split_buckets["train"].append(rail_pairs[idx])
    for idx in val_idx:
        split_buckets["val"].append(rail_pairs[idx])
    for idx in test_idx:
        split_buckets["test"].append(rail_pairs[idx])

    for split, pairs in split_buckets.items():
        for image_path, label_path in tqdm(pairs, desc=f"Copy rail {split}"):
            _copy_pair_to_split(image_path, label_path, split, images_root, labels_root, prefix="rail_")

    for item in tqdm(audio_entries, desc="Convert audio"):
        split = item["split"]
        class_id = item["class_id"]
        wav_path = item["wav_path"]
        base_name = f"audio_{wav_path.stem}"

        image_dst = images_root / split / f"{base_name}.png"
        label_dst = labels_root / split / f"{base_name}.txt"
        ensure_dir(image_dst.parent)
        ensure_dir(label_dst.parent)
        if not image_dst.exists():
            _generate_spectrogram(wav_path, image_dst)
        _write_yolo_label(label_dst, class_id, 0.5, 0.5, 1.0, 1.0)

    names = rail_names + audio_names
    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(names)},
    }
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    LOGGER.info("YOLO data.yaml created at: %s", yaml_path)
    return yaml_path

