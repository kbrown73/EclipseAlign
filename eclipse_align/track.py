from __future__ import annotations

import numpy as np

from .models import FrameDetection


def estimate_common_radius(detections: list[FrameDetection]) -> float | None:
    radii = [
        d.radius
        for d in detections
        if d.radius is not None and d.confidence >= 0.35 and d.status in {"ok", "clipped"}
    ]
    if not radii:
        return None
    return float(np.median(np.asarray(radii, dtype=np.float64)))


def flag_radius_outliers(
    detections: list[FrameDetection],
    common_radius: float | None,
    *,
    tolerance: float = 0.30,
) -> None:
    if common_radius is None:
        return
    for detection in detections:
        if detection.radius is None:
            continue
        error = abs(detection.radius - common_radius) / max(common_radius, 1.0)
        if error <= tolerance:
            continue
        if "radius_outlier" not in detection.flags:
            detection.flags.append("radius_outlier")
        if detection.status == "ok":
            detection.status = "low_confidence"
        detection.confidence = min(detection.confidence, 0.34)


def fill_missing_centers(detections: list[FrameDetection], common_radius: float | None) -> None:
    indexes = np.arange(len(detections), dtype=np.float64)
    good_indexes = []
    xs = []
    ys = []
    for idx, detection in enumerate(detections):
        if detection.has_center and detection.status in {"ok", "clipped", "low_confidence"}:
            good_indexes.append(idx)
            xs.append(detection.center_x)
            ys.append(detection.center_y)

    if len(good_indexes) < 2:
        return

    good_indexes_array = np.asarray(good_indexes, dtype=np.float64)
    interp_x = np.interp(indexes, good_indexes_array, np.asarray(xs, dtype=np.float64))
    interp_y = np.interp(indexes, good_indexes_array, np.asarray(ys, dtype=np.float64))

    for idx, detection in enumerate(detections):
        if detection.has_center and detection.status != "failed":
            continue
        detection.center_x = float(interp_x[idx])
        detection.center_y = float(interp_y[idx])
        if detection.radius is None:
            detection.radius = common_radius
        detection.status = "estimated"
        detection.confidence = min(detection.confidence, 0.25)
        if "interpolated" not in detection.flags:
            detection.flags.append("interpolated")


def assign_translations(
    detections: list[FrameDetection],
    *,
    target_x: float | None = None,
    target_y: float | None = None,
) -> None:
    for detection in detections:
        if not detection.has_center:
            continue
        tx = target_x if target_x is not None else detection.width / 2.0
        ty = target_y if target_y is not None else detection.height / 2.0
        detection.translation_x = float(tx - detection.center_x)
        detection.translation_y = float(ty - detection.center_y)


def refine_detections(detections: list[FrameDetection]) -> float | None:
    common_radius = estimate_common_radius(detections)
    flag_radius_outliers(detections, common_radius)
    fill_missing_centers(detections, common_radius)
    assign_translations(detections)
    return common_radius
