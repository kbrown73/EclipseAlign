from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detect import DetectionConfig, luminance, normalize_luminance
from .exr_io import read_exr, write_exr
from .models import FrameDetection
from .render import translate_image


@dataclass
class PolishConfig:
    max_shift_px: float = 2.0
    reference_radius: int = 2
    crop_scale: float = 3.0
    min_response: float = 0.12


@dataclass
class PolishResult:
    frame_index: int
    filename: str
    segment_id: int
    residual_dx: float
    residual_dy: float
    confidence: float
    score: float
    source: str


def polish_aligned_outputs(
    output_dir: Path,
    detections: list[FrameDetection],
    *,
    config: PolishConfig | None = None,
) -> list[PolishResult]:
    config = config or PolishConfig()
    crops = [polish_crop(output_dir / detection.filename, detection, config) for detection in detections]
    results = estimate_polish_offsets(crops, detections, config)
    for result in results:
        if result.source != "phase_correlation":
            continue
        path = output_dir / result.filename
        image = read_exr(path)
        polished = translate_image(image, result.residual_dx, result.residual_dy)
        write_exr(path, polished)
    return results


def estimate_polish_offsets(
    crops: list[np.ndarray | None],
    detections: list[FrameDetection],
    config: PolishConfig,
) -> list[PolishResult]:
    results = []
    for idx, detection in enumerate(detections):
        crop = crops[idx]
        if crop is None:
            results.append(no_polish_result(idx, detection, "no_crop"))
            continue
        reference = local_reference(crops, detections, idx, config)
        if reference is None:
            results.append(no_polish_result(idx, detection, "no_reference"))
            continue
        shift, response = cv2.phaseCorrelate(reference, crop)
        dx = -float(shift[0])
        dy = -float(shift[1])
        magnitude = float(np.hypot(dx, dy))
        if response < config.min_response or magnitude > config.max_shift_px:
            results.append(
                PolishResult(
                    frame_index=idx,
                    filename=detection.filename,
                    segment_id=detection.segment_id,
                    residual_dx=0.0,
                    residual_dy=0.0,
                    confidence=float(max(0.0, min(response, 1.0))),
                    score=magnitude,
                    source="rejected",
                )
            )
            continue
        results.append(
            PolishResult(
                frame_index=idx,
                filename=detection.filename,
                segment_id=detection.segment_id,
                residual_dx=dx,
                residual_dy=dy,
                confidence=float(max(0.0, min(response, 1.0))),
                score=magnitude,
                source="phase_correlation",
            )
        )
    return results


def polish_crop(path: Path, detection: FrameDetection, config: PolishConfig) -> np.ndarray | None:
    if not path.exists() or detection.radius is None:
        return None
    image = read_exr(path)
    height, width = image.shape[:2]
    radius = detection.radius
    side = int(max(96, min(min(width, height), round(radius * config.crop_scale))))
    cx = width / 2.0
    cy = height / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    crop = image[top : top + side, left : left + side]
    if crop.shape[0] != side or crop.shape[1] != side:
        return None
    preview = normalize_luminance(luminance(crop), DetectionConfig())
    smoothed = cv2.GaussianBlur(preview, (0, 0), sigmaX=2.0)
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    signal = np.hypot(grad_x, grad_y).astype(np.float32)
    mask = circular_mask(signal.shape, radius=min(side / 2.0, radius * 1.15))
    signal = signal * mask
    signal = (signal - float(np.mean(signal[mask > 0]))) * mask
    window = cv2.createHanningWindow((side, side), cv2.CV_32F)
    signal = signal * window
    norm = float(np.linalg.norm(signal))
    if norm <= 1e-9:
        return None
    return (signal / norm).astype(np.float32)


def local_reference(
    crops: list[np.ndarray | None],
    detections: list[FrameDetection],
    frame_index: int,
    config: PolishConfig,
) -> np.ndarray | None:
    detection = detections[frame_index]
    candidates = []
    start = max(0, frame_index - config.reference_radius)
    stop = min(len(crops), frame_index + config.reference_radius + 1)
    for idx in range(start, stop):
        if idx == frame_index:
            continue
        if detections[idx].segment_id != detection.segment_id:
            continue
        crop = crops[idx]
        if crop is not None:
            candidates.append(crop)
    if not candidates:
        return None
    reference = np.median(np.stack(candidates, axis=0), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(reference))
    if norm <= 1e-9:
        return None
    return (reference / norm).astype(np.float32)


def no_polish_result(frame_index: int, detection: FrameDetection, source: str) -> PolishResult:
    return PolishResult(
        frame_index=frame_index,
        filename=detection.filename,
        segment_id=detection.segment_id,
        residual_dx=0.0,
        residual_dy=0.0,
        confidence=0.0,
        score=0.0,
        source=source,
    )


def circular_mask(shape: tuple[int, int], radius: float) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    edge = np.clip((radius - dist) / max(radius * 0.05, 1.0), 0.0, 1.0)
    return edge.astype(np.float32)
