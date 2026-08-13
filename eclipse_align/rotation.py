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
    drift_min_points: int = 8
    drift_min_distance_px: float = 180.0
    drift_min_confidence: float = 0.5
    drift_jump_threshold_degrees: float = 4.0
    drift_expected_window: int = 2
    registration_fallback: bool = True
    boundary_offset: int = 1
    crop_scale: float = 3.2
    crop_max_size: int = 1200
    search_degrees: float = 8.0
    coarse_step_degrees: float = 0.5
    fine_step_degrees: float = 0.1
    min_zero_score_improvement: float = 0.02
    silhouette_min_area_fraction: float = 0.80
    silhouette_min_delta_degrees: float = 0.8
    silhouette_edge_agreement_degrees: float = 1.5
    min_boundary_confidence: float = 0.6


@dataclass
class BoundaryRotation:
    boundary_index: int
    from_segment: int
    to_segment: int
    delta_deg: float
    confidence: float
    score: float
    source: str = "registration"
    from_drift_angle_deg: float | None = None
    to_drift_angle_deg: float | None = None
    raw_drift_delta_deg: float | None = None
    expected_drift_delta_deg: float | None = None


@dataclass
class SegmentDrift:
    segment_id: int
    first_frame: int
    last_frame: int
    points: int
    distance_px: float
    angle_deg: float
    confidence: float


@dataclass
class SilhouetteEstimate:
    angle_deg: float
    area_fraction: float
    centroid_distance_fraction: float


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


def angle_delta_degrees(to_angle: float, from_angle: float) -> float:
    return float((to_angle - from_angle + 180.0) % 360.0 - 180.0)


def robust_motion_fit(
    frame_indices: np.ndarray,
    centers_x: np.ndarray,
    centers_y: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    mask = np.ones(len(frame_indices), dtype=bool)
    slope_x = 0.0
    slope_y = 0.0
    for _ in range(4):
        if int(np.sum(mask)) < 2:
            break
        slope_x, intercept_x = np.polyfit(frame_indices[mask], centers_x[mask], 1)
        slope_y, intercept_y = np.polyfit(frame_indices[mask], centers_y[mask], 1)
        predicted_x = slope_x * frame_indices + intercept_x
        predicted_y = slope_y * frame_indices + intercept_y
        residuals = np.hypot(centers_x - predicted_x, centers_y - predicted_y)
        active_residuals = residuals[mask]
        median = float(np.median(active_residuals))
        mad = float(np.median(np.abs(active_residuals - median)))
        sigma = 1.4826 * mad
        threshold = max(8.0, median + 3.0 * sigma)
        next_mask = residuals <= threshold
        if int(np.sum(next_mask)) < 2 or np.array_equal(next_mask, mask):
            break
        mask = next_mask
    return float(slope_x), float(slope_y), mask


def estimate_segment_drifts(
    detections: list[FrameDetection],
    config: RotationConfig,
) -> dict[int, SegmentDrift]:
    by_segment: dict[int, list[tuple[int, FrameDetection]]] = {}
    for idx, detection in enumerate(detections):
        if not detection.has_center:
            continue
        if detection.status == "failed":
            continue
        if detection.confidence < config.drift_min_confidence:
            continue
        by_segment.setdefault(detection.segment_id, []).append((idx, detection))

    drifts: dict[int, SegmentDrift] = {}
    for segment_id, items in by_segment.items():
        if len(items) < config.drift_min_points:
            continue
        frame_indices = np.asarray([idx for idx, _ in items], dtype=np.float64)
        centers_x = np.asarray([d.center_x for _, d in items], dtype=np.float64)
        centers_y = np.asarray([d.center_y for _, d in items], dtype=np.float64)
        slope_x, slope_y, inlier_mask = robust_motion_fit(frame_indices, centers_x, centers_y)
        inlier_count = int(np.sum(inlier_mask))
        if inlier_count < config.drift_min_points:
            continue
        frame_span = float(frame_indices[inlier_mask][-1] - frame_indices[inlier_mask][0])
        distance = float(np.hypot(slope_x, slope_y) * frame_span)
        if distance <= 1e-9:
            continue
        confidence = min(1.0, distance / config.drift_min_distance_px)
        if confidence < config.min_boundary_confidence:
            continue
        angle = float(np.degrees(np.arctan2(slope_y, slope_x)))
        drifts[segment_id] = SegmentDrift(
            segment_id=segment_id,
            first_frame=int(frame_indices[inlier_mask][0]),
            last_frame=int(frame_indices[inlier_mask][-1]),
            points=inlier_count,
            distance_px=distance,
            angle_deg=angle,
            confidence=confidence,
        )
    return drifts


def typical_delta_limit(deltas: list[float], config: RotationConfig) -> tuple[float, float]:
    if not deltas:
        return 0.0, config.drift_jump_threshold_degrees
    values = np.asarray(deltas, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    return median, max(config.drift_jump_threshold_degrees, 3.0 * sigma)


def expected_drift_delta(
    deltas: list[float],
    index: int,
    config: RotationConfig,
) -> float:
    median, limit = typical_delta_limit(deltas, config)
    start = max(0, index - config.drift_expected_window)
    stop = min(len(deltas), index + config.drift_expected_window + 1)
    local = [delta for delta in deltas[start:stop] if abs(delta - median) <= limit]
    if local:
        return float(np.median(np.asarray(local, dtype=np.float64)))
    normal = [delta for delta in deltas if abs(delta - median) <= limit]
    if normal:
        return float(np.median(np.asarray(normal, dtype=np.float64)))
    return median


def drift_boundary_rotations(
    boundaries: list[int],
    drifts: dict[int, SegmentDrift],
    config: RotationConfig,
) -> list[BoundaryRotation]:
    max_segment = max(drifts) if drifts else -1
    raw_deltas: list[float | None] = []
    comparable_deltas: list[float] = []
    for segment_id in range(max_segment):
        before = drifts.get(segment_id)
        after = drifts.get(segment_id + 1)
        if before is None or after is None:
            raw_deltas.append(None)
            continue
        raw_delta = angle_delta_degrees(after.angle_deg, before.angle_deg)
        raw_deltas.append(raw_delta)
        comparable_deltas.append(raw_delta)

    results: list[BoundaryRotation] = []
    comparable_index = 0
    for boundary, to_segment in zip(boundaries, range(1, len(boundaries) + 1)):
        from_segment = to_segment - 1
        before = drifts.get(from_segment)
        after = drifts.get(to_segment)
        if before is None or after is None:
            results.append(
                BoundaryRotation(
                    boundary_index=boundary,
                    from_segment=from_segment,
                    to_segment=to_segment,
                    delta_deg=0.0,
                    confidence=0.0,
                    score=0.0,
                    source="drift_unavailable",
                )
            )
            continue

        raw_delta = raw_deltas[from_segment]
        if raw_delta is None:
            expected_delta = 0.0
        else:
            expected_delta = expected_drift_delta(comparable_deltas, comparable_index, config)
            comparable_index += 1
        excess_delta = float(raw_delta or 0.0) - expected_delta
        if abs(excess_delta) < config.drift_jump_threshold_degrees:
            correction_delta = 0.0
        else:
            correction_delta = -excess_delta
        confidence = min(before.confidence, after.confidence)
        results.append(
            BoundaryRotation(
                boundary_index=boundary,
                from_segment=from_segment,
                to_segment=to_segment,
                delta_deg=correction_delta,
                confidence=confidence,
                score=abs(excess_delta),
                source="drift",
                from_drift_angle_deg=before.angle_deg,
                to_drift_angle_deg=after.angle_deg,
                raw_drift_delta_deg=float(raw_delta or 0.0),
                expected_drift_delta_deg=expected_delta,
            )
        )
    return results


def registration_boundary_tasks(
    inputs: list[Path],
    detections: list[FrameDetection],
    boundaries: list[int],
    config: RotationConfig,
    boundary_results: list[BoundaryRotation],
) -> list[tuple[int, tuple[Path, FrameDetection, Path, FrameDetection, RotationConfig]]]:
    tasks = []
    for result, boundary in zip(boundary_results, boundaries):
        before_idx = max(0, boundary - config.boundary_offset)
        after_idx = min(len(detections) - 1, boundary + config.boundary_offset - 1)
        tasks.append(
            (
                result.to_segment,
                (
                    inputs[before_idx],
                    detections[before_idx],
                    inputs[after_idx],
                    detections[after_idx],
                    config,
                ),
            )
        )
    return tasks


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

    drifts = estimate_segment_drifts(detections, config)
    boundary_results = drift_boundary_rotations(boundaries, drifts, config)
    tasks = []
    if config.registration_fallback and inputs:
        tasks = registration_boundary_tasks(inputs, detections, boundaries, config, boundary_results)
    if tasks:
        payloads = [task for _, task in tasks]
        if map_fn is None:
            registration_results = [estimate_boundary_rotation(task) for task in payloads]
        else:
            registration_results = map_fn(estimate_boundary_rotation, payloads, desc="rotation")
        by_segment = dict(zip((segment_id for segment_id, _ in tasks), registration_results))
        combined_results = []
        for result in boundary_results:
            registration_result = by_segment.get(result.to_segment)
            if (
                registration_result is not None
                and registration_result.confidence >= config.min_boundary_confidence
            ):
                registration_result.from_drift_angle_deg = result.from_drift_angle_deg
                registration_result.to_drift_angle_deg = result.to_drift_angle_deg
                registration_result.raw_drift_delta_deg = result.raw_drift_delta_deg
                registration_result.expected_drift_delta_deg = result.expected_drift_delta_deg
                combined_results.append(registration_result)
            else:
                combined_results.append(result)
        boundary_results = combined_results

    segment_rotations = {0: 0.0}
    segment_sources = {0: "none"}
    for result in boundary_results:
        previous_rotation = segment_rotations.get(result.from_segment, 0.0)
        previous_source = segment_sources.get(result.from_segment, "none")
        delta = result.delta_deg if result.confidence >= config.min_boundary_confidence else 0.0
        next_rotation = previous_rotation + delta
        segment_rotations[result.to_segment] = next_rotation
        if abs(delta) > 1e-9:
            segment_sources[result.to_segment] = result.source
        elif abs(next_rotation) > 1e-9:
            segment_sources[result.to_segment] = previous_source
        else:
            segment_sources[result.to_segment] = "none"

    for detection in detections:
        detection.rotation_deg = float(segment_rotations.get(detection.segment_id, 0.0))
        detection.rotation_source = (
            segment_sources.get(detection.segment_id, "none")
            if abs(detection.rotation_deg) > 1e-9
            else "none"
        )
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
    texture = estimate_boundary_rotation_with_mode(
        before_path,
        before,
        after_path,
        after,
        config,
        mode="texture",
        source="registration",
    )
    if texture.confidence >= config.min_boundary_confidence:
        return texture

    edge = estimate_boundary_rotation_with_mode(
        before_path,
        before,
        after_path,
        after,
        config,
        mode="edge",
        source="edge_registration",
    )
    if edge.confidence >= config.min_boundary_confidence:
        return edge

    silhouette = estimate_silhouette_boundary_rotation(before_path, before, after_path, after, edge, config)
    if silhouette.confidence >= config.min_boundary_confidence:
        return silhouette
    return texture if texture.confidence >= edge.confidence else edge


def estimate_boundary_rotation_with_mode(
    before_path: Path,
    before: FrameDetection,
    after_path: Path,
    after: FrameDetection,
    config: RotationConfig,
    *,
    mode: str,
    source: str,
) -> BoundaryRotation:
    before_crop = aligned_registration_crop(before_path, before, config, mode=mode)
    after_crop = aligned_registration_crop(after_path, after, config, mode=mode)
    delta, score, next_score = search_rotation(before_crop, after_crop, config)
    zero_score = score_angles(before_crop, after_crop, np.asarray([0.0]))[0][1]
    improvement = score - zero_score
    if improvement < config.min_zero_score_improvement:
        delta = 0.0
        confidence = 0.0
    else:
        confidence = registration_confidence(score, next_score, zero_score)
    return BoundaryRotation(
        boundary_index=after.segment_id,
        from_segment=before.segment_id,
        to_segment=after.segment_id,
        delta_deg=delta,
        confidence=confidence,
        score=score,
        source=source,
    )


def estimate_silhouette_boundary_rotation(
    before_path: Path,
    before: FrameDetection,
    after_path: Path,
    after: FrameDetection,
    edge_result: BoundaryRotation,
    config: RotationConfig,
) -> BoundaryRotation:
    before_estimate = estimate_silhouette_angle(before_path, before)
    after_estimate = estimate_silhouette_angle(after_path, after)
    if before_estimate is None or after_estimate is None:
        confidence = 0.0
        delta = 0.0
    else:
        delta = angle_delta_degrees(after_estimate.angle_deg, before_estimate.angle_deg)
        area_fraction = min(before_estimate.area_fraction, after_estimate.area_fraction)
        agrees_with_edge = abs(angle_delta_degrees(delta, edge_result.delta_deg)) <= (
            config.silhouette_edge_agreement_degrees
        )
        if (
            area_fraction >= config.silhouette_min_area_fraction
            and abs(delta) >= config.silhouette_min_delta_degrees
            and agrees_with_edge
        ):
            confidence = min(
                1.0,
                0.65 + min(0.25, abs(delta) / 10.0) + min(0.10, max(0.0, area_fraction - 0.80)),
            )
        else:
            confidence = 0.0
            delta = 0.0
    return BoundaryRotation(
        boundary_index=after.segment_id,
        from_segment=before.segment_id,
        to_segment=after.segment_id,
        delta_deg=delta,
        confidence=confidence,
        score=abs(delta),
        source="silhouette",
    )


def aligned_registration_crop(
    path: Path,
    detection: FrameDetection,
    config: RotationConfig,
    *,
    mode: str = "texture",
) -> np.ndarray:
    crop, radius = aligned_luminance_crop(path, detection, config.crop_scale, config.crop_max_size)
    preview = normalize_luminance(luminance(crop), DetectionConfig())
    if mode == "edge":
        smoothed = cv2.GaussianBlur(preview, (0, 0), sigmaX=2.0)
        grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        preview = np.hypot(grad_x, grad_y).astype(np.float32)
    else:
        blur = cv2.GaussianBlur(preview, (0, 0), sigmaX=max(4.0, radius * 0.06))
        preview = preview - blur
    mask = circular_mask(preview.shape, radius=min(min(preview.shape) / 2.0, radius * 1.08))
    preview = preview * mask
    preview = (preview - float(np.mean(preview[mask > 0]))) * mask
    norm = float(np.linalg.norm(preview))
    if norm > 0:
        preview = preview / norm
    return preview.astype(np.float32)


def aligned_luminance_crop(
    path: Path,
    detection: FrameDetection,
    crop_scale: float,
    crop_max_size: int,
) -> tuple[np.ndarray, float]:
    image = read_exr(path)
    if detection.translation_x is None or detection.translation_y is None:
        raise ValueError(f"Detection has no translation: {detection.filename}")
    aligned = translate_image(image, detection.translation_x, detection.translation_y)
    height, width = aligned.shape[:2]
    radius = detection.radius or min(width, height) / 8.0
    side = int(min(crop_max_size, max(128, round(radius * crop_scale))))
    cx = width / 2.0
    cy = height / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    return aligned[top : top + side, left : left + side], radius


def estimate_silhouette_angle(path: Path, detection: FrameDetection) -> SilhouetteEstimate | None:
    crop, radius = aligned_luminance_crop(path, detection, crop_scale=2.4, crop_max_size=900)
    preview = normalize_luminance(luminance(crop), DetectionConfig())
    height, width = preview.shape
    yy, xx = np.ogrid[:height, :width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disk = dist <= radius * 0.98
    values = preview[disk]
    if values.size == 0:
        return None
    threshold = max(0.10, min(0.45, float(np.percentile(values, 25))))
    dark = ((preview < threshold) & disk).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    labels, _, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    components = [
        (stats[idx, cv2.CC_STAT_AREA], centroids[idx])
        for idx in range(1, labels)
        if stats[idx, cv2.CC_STAT_AREA] >= 50
    ]
    if not components:
        return None
    area, centroid = max(components, key=lambda item: item[0])
    dx = float(centroid[0] - cx)
    dy = float(centroid[1] - cy)
    disk_area = max(1, int(np.sum(disk)))
    return SilhouetteEstimate(
        angle_deg=float(np.degrees(np.arctan2(dy, dx))),
        area_fraction=float(area / disk_area),
        centroid_distance_fraction=float(np.hypot(dx, dy) / max(radius, 1e-6)),
    )


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


def registration_confidence(best_score: float, next_score: float, zero_score: float) -> float:
    if best_score <= 0:
        return 0.0
    score_scale = max(abs(best_score), 1e-6)
    zero_improvement = max(0.0, best_score - zero_score) / score_scale
    peak_separation = max(0.0, best_score - next_score) / score_scale
    return float(np.clip(zero_improvement * 5.0 + peak_separation * 8.0, 0.0, 1.0))
