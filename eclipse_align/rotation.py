from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .detect import DetectionConfig, luminance, normalize_luminance
from .exr_io import read_exr
from .models import FrameDetection
from .render import translate_image


@dataclass
class RotationConfig:
    jump_threshold_px: float = 120.0
    min_segment_length: int = 8
    boundary_offset: int = 2
    crop_scale: float = 3.2
    crop_max_size: int = 1200
    search_degrees: float = 8.0
    coarse_step_degrees: float = 0.5
    fine_step_degrees: float = 0.1
    min_zero_score_improvement: float = 0.02
    min_boundary_confidence: float = 0.6


@dataclass
class BoundaryRotation:
    boundary_index: int
    from_segment: int
    to_segment: int
    delta_deg: float
    confidence: float
    score: float


def detect_reframe_boundaries(
    detections: list[FrameDetection],
    *,
    threshold_px: float,
    min_segment_length: int = 1,
) -> list[int]:
    boundaries: list[int] = []
    last_boundary = 0
    for idx in range(1, len(detections)):
        prev = detections[idx - 1]
        cur = detections[idx]
        if not prev.has_center or not cur.has_center:
            continue
        delta = float(np.hypot(cur.center_x - prev.center_x, cur.center_y - prev.center_y))
        if delta >= threshold_px:
            if idx - last_boundary < min_segment_length:
                if boundaries:
                    boundaries[-1] = idx
                    last_boundary = idx
                continue
            boundaries.append(idx)
            last_boundary = idx
    return boundaries


def assign_segments(detections: list[FrameDetection], boundaries: list[int]) -> None:
    boundary_set = set(boundaries)
    segment_id = 0
    for idx, detection in enumerate(detections):
        if idx in boundary_set:
            segment_id += 1
        detection.segment_id = segment_id


def initialize_rotation_metadata(detections: list[FrameDetection]) -> None:
    for detection in detections:
        detection.rotation_deg = 0.0
        detection.rotation_confidence = 0.0
        detection.rotation_source = "none"


def estimate_rotations(
    inputs: list[Path],
    detections: list[FrameDetection],
    *,
    config: RotationConfig | None = None,
    map_fn: Callable | None = None,
) -> list[BoundaryRotation]:
    config = config or RotationConfig()
    boundaries = detect_reframe_boundaries(
        detections,
        threshold_px=config.jump_threshold_px,
        min_segment_length=config.min_segment_length,
    )
    assign_segments(detections, boundaries)
    initialize_rotation_metadata(detections)
    if not boundaries:
        return []

    tasks = []
    for boundary in boundaries:
        before_idx = max(0, boundary - config.boundary_offset)
        after_idx = min(len(detections) - 1, boundary + config.boundary_offset - 1)
        before = detections[before_idx]
        after = detections[after_idx]
        if before.rotation_confidence == 0.0:
            before.rotation_confidence = 1.0
        tasks.append(
            (
                inputs[before_idx],
                before,
                inputs[after_idx],
                after,
                config,
            )
        )

    if map_fn is None:
        boundary_results = [estimate_boundary_rotation(task) for task in tasks]
    else:
        boundary_results = map_fn(estimate_boundary_rotation, tasks, desc="rotation")

    segment_rotations = {0: 0.0}
    for result in boundary_results:
        previous_rotation = segment_rotations.get(result.from_segment, 0.0)
        delta = result.delta_deg if result.confidence >= config.min_boundary_confidence else 0.0
        segment_rotations[result.to_segment] = previous_rotation + delta

    for detection in detections:
        detection.rotation_deg = float(segment_rotations.get(detection.segment_id, 0.0))
        detection.rotation_source = "registration" if abs(detection.rotation_deg) > 1e-9 else "none"
        if detection.segment_id == 0:
            detection.rotation_confidence = 1.0
        else:
            confidences = [
                result.confidence
                for result in boundary_results
                if result.to_segment <= detection.segment_id
                and result.confidence >= config.min_boundary_confidence
            ]
            detection.rotation_confidence = float(min(confidences)) if confidences else 0.0
    return boundary_results


def estimate_boundary_rotation(
    payload: tuple[Path, FrameDetection, Path, FrameDetection, RotationConfig],
) -> BoundaryRotation:
    before_path, before, after_path, after, config = payload
    before_crop = aligned_registration_crop(before_path, before, config)
    after_crop = aligned_registration_crop(after_path, after, config)
    delta, score, next_score = search_rotation(before_crop, after_crop, config)
    zero_score = score_angles(before_crop, after_crop, np.asarray([0.0]))[0][1]
    improvement = score - zero_score
    if improvement < config.min_zero_score_improvement:
        delta = 0.0
        confidence = 0.0
    else:
        confidence = rotation_confidence(score, next_score)
    return BoundaryRotation(
        boundary_index=after.segment_id,
        from_segment=before.segment_id,
        to_segment=after.segment_id,
        delta_deg=delta,
        confidence=confidence,
        score=score,
    )


def aligned_registration_crop(
    path: Path,
    detection: FrameDetection,
    config: RotationConfig,
) -> np.ndarray:
    image = read_exr(path)
    if detection.translation_x is None or detection.translation_y is None:
        raise ValueError(f"Detection has no translation: {detection.filename}")
    aligned = translate_image(image, detection.translation_x, detection.translation_y)
    height, width = aligned.shape[:2]
    radius = detection.radius or min(width, height) / 8.0
    side = int(min(config.crop_max_size, max(128, round(radius * config.crop_scale))))
    cx = width / 2.0
    cy = height / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    crop = aligned[top : top + side, left : left + side]
    preview = normalize_luminance(luminance(crop), DetectionConfig())
    mask = circular_mask(preview.shape, radius=min(side, radius * 1.35))
    preview = (preview - float(np.mean(preview[mask > 0]))) * mask
    norm = float(np.linalg.norm(preview))
    if norm > 0:
        preview = preview / norm
    return preview.astype(np.float32)


def circular_mask(shape: tuple[int, int], radius: float) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (dist <= radius).astype(np.float32)
    edge = np.clip((radius - dist) / max(radius * 0.05, 1.0), 0.0, 1.0)
    return np.minimum(mask, edge).astype(np.float32)


def search_rotation(
    reference: np.ndarray,
    moving: np.ndarray,
    config: RotationConfig,
) -> tuple[float, float, float]:
    coarse_angles = np.arange(
        -config.search_degrees,
        config.search_degrees + config.coarse_step_degrees * 0.5,
        config.coarse_step_degrees,
    )
    coarse = score_angles(reference, moving, coarse_angles)
    best_angle = coarse[0][0]
    fine_angles = np.arange(
        best_angle - config.coarse_step_degrees,
        best_angle + config.coarse_step_degrees + config.fine_step_degrees * 0.5,
        config.fine_step_degrees,
    )
    fine = score_angles(reference, moving, fine_angles)
    best = fine[0]
    next_best = fine[1] if len(fine) > 1 else best
    return float(best[0]), float(best[1]), float(next_best[1])


def score_angles(
    reference: np.ndarray,
    moving: np.ndarray,
    angles: np.ndarray,
) -> list[tuple[float, float]]:
    scores = []
    height, width = reference.shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    for angle in angles:
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            moving,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        score = float(np.sum(reference * rotated))
        scores.append((float(angle), score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def rotation_confidence(best_score: float, next_score: float) -> float:
    if best_score <= 0:
        return 0.0
    separation = max(0.0, best_score - next_score)
    return float(np.clip(0.25 + separation / max(abs(best_score), 1e-6) * 20.0, 0.0, 1.0))
