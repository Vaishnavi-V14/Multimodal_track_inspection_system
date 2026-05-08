from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    rail_url: str | None = (
        "https://huggingface.co/datasets/sleepysaurus/RailRakshak-Track-Detection-Data/resolve/main/Rail-DB.zip"
    )
    rail_local_dir: Path = Path("dataset/rail5k")
    audio_url: str = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
    raw_root: Path = Path("data/raw")
    processed_root: Path = Path("data/processed")


@dataclass
class TrainConfig:
    model: str = "yolov26n.pt"
    epochs: int = 20
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"
    project: str = "runs/train"
    name: str = "multimodal_rail_audio"

