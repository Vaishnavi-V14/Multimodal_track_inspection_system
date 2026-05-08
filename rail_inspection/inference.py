"""YOLO inference with industry-oriented rail/crack preprocessing and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from ultralytics import YOLO


_CRACK_KEYWORDS = ("crack", "fissure", "split", "fracture", "broken")


@dataclass
class DetectionRecord:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    severity_score: float = 0.0
    texture_score: float = 0.0
    frame_index: int | None = None


def load_model(model_path: str | Path) -> YOLO:
    return YOLO(str(model_path))


def preprocess_track_frame(
    image_bgr: np.ndarray,
    *,
    clahe: bool = True,
    bilateral_denoise: bool = True,
    sharpen: bool = False,
) -> np.ndarray:
    """Enhance low-contrast rail surfaces for crack visibility."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    if clahe:
        clahe_op = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_ch = clahe_op.apply(l_ch)
    lab_merged = cv2.merge([l_ch, a_ch, b_ch])
    out = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    if bilateral_denoise:
        out = cv2.bilateralFilter(out, d=5, sigmaColor=50, sigmaSpace=50)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(out, -1, kernel)
        out = cv2.addWeighted(out, 0.65, sharpened, 0.35, 0)
    return out


def _roi_texture_score(image_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    h, w = image_bgr.shape[:2]
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w, int(x2)), min(h, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    roi = image_bgr[yi1:yi2, xi1:xi2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(np.mean(edges > 0))
    tex = min(lap_var / 800.0, 1.0) * 0.55 + min(edge_density * 4.0, 1.0) * 0.45
    return float(min(max(tex, 0.0), 1.0))


def _crack_class_bonus(class_name: str) -> float:
    lowered = class_name.lower()
    return 0.12 if any(k in lowered for k in _CRACK_KEYWORDS) else 0.0


def _aspect_crack_bonus(x1: float, y1: float, x2: float, y2: float) -> float:
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    ar = bw / bh if bh >= bw else bh / bw
    return min(max(ar / 12.0, 0.0), 0.08)


def compute_severity_score(
    record: DetectionRecord,
    image_shape: tuple[int, int, int],
    *,
    weight_conf: float = 0.45,
    weight_tex: float = 0.35,
    weight_bonus: float = 0.20,
) -> float:
    h, w = image_shape[0], image_shape[1]
    area = max((record.x2 - record.x1) * (record.y2 - record.y1), 1.0)
    frame_area = float(max(h * w, 1))
    area_norm = min(area / frame_area, 1.0)
    bonus = _crack_class_bonus(record.class_name) + _aspect_crack_bonus(record.x1, record.y1, record.x2, record.y2)
    score = (
        weight_conf * record.confidence
        + weight_tex * record.texture_score
        + weight_bonus * (bonus + area_norm * 0.5)
    )
    return float(min(max(score, 0.0), 1.0))


def _extract_records(result: Any, image_bgr: np.ndarray, frame_index: int | None = None) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    if result.boxes is None:
        return records

    names = result.names
    xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else np.empty((0, 4))
    conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.empty((0,))
    cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.empty((0,), dtype=int)

    for i in range(len(xyxy)):
        class_id = int(cls[i])
        rec = DetectionRecord(
            class_name=str(names.get(class_id, f"class_{class_id}")),
            confidence=float(conf[i]),
            x1=float(xyxy[i][0]),
            y1=float(xyxy[i][1]),
            x2=float(xyxy[i][2]),
            y2=float(xyxy[i][3]),
            texture_score=_roi_texture_score(image_bgr, xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3]),
            frame_index=frame_index,
        )
        rec.severity_score = compute_severity_score(rec, image_bgr.shape)
        records.append(rec)
    return records


def predict_image(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float = 0.25,
    *,
    preprocess: bool = False,
    clahe: bool = True,
    bilateral: bool = True,
    sharpen: bool = False,
    frame_index: int | None = None,
    iou: float = 0.55,
    imgsz: int | None = None,
) -> tuple[np.ndarray, list[DetectionRecord]]:
    proc = (
        preprocess_track_frame(image_bgr, clahe=clahe, bilateral_denoise=bilateral, sharpen=sharpen)
        if preprocess
        else image_bgr
    )
    kw: dict[str, Any] = {"source": proc, "conf": conf, "verbose": False, "iou": iou}
    if imgsz is not None:
        kw["imgsz"] = imgsz
    results = model.predict(**kw)
    result = results[0]
    plotted = result.plot()
    records = _extract_records(result, proc, frame_index=frame_index)
    return plotted, records


def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unable to decode image file.")
    return image_bgr


def iter_video_frames(path: str | Path, *, frame_stride: int = 1, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    idx = 0
    emitted = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(frame_stride, 1) == 0:
                yield idx, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
