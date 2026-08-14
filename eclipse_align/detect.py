from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .exr_io import read_exr
from .models import FrameDetection


@dataclass
class DetectionConfig:
    work_max_dim: int = 1400
    threshold: float = 0.18
    low_percentile: float = 50.0
    high_percentile: float = 99.95
    min_component_area_fraction: float = 0.00002
    ransac_iterations: int = 900
    ransac_tolerance_fraction: float = 0.018
    max_edge_points: int = 3500
    min_limb_points: int = 48
    seed: int = 1234
    distorted_limb_median_px: float = 6.0
    distorted_limb_support_fraction: float = 0.52


def luminance(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 1:
        return image[:, :, 0].astype(np.float32, copy=False)
    rgb = image[:, :, :3].astype(np.float32, copy=False)
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def normalize_luminance(luma: np.ndarray, config: DetectionConfig) -> np.ndarray:
    finite = luma[np.isfinite(luma)]
    if finite.size == 0:
        return np.zeros_like(luma, dtype=np.float32)
    low = float(np.percentile(finite, config.low_percentile))
    high = float(np.percentile(finite, config.high_percentile))
    if high <= low:
        high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(luma, dtype=np.float32)
    normalized = (luma - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def downsample_for_detection(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    largest = max(width, height)
    if largest <= max_dim:
        return image, 1.0
    scale = max_dim / largest
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def make_mask(normalized: np.ndarray, config: DetectionConfig) -> np.ndarray:
    mask = (normalized >= config.threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def largest_plausible_component(mask: np.ndarray, config: DetectionConfig) -> np.ndarray | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    min_area = mask.shape[0] * mask.shape[1] * config.min_component_area_fraction
    components: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    if not components:
        return None
    _, label = max(components)
    return (labels == label).astype(np.uint8) * 255


def circle_from_three(points: np.ndarray) -> tuple[float, float, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = points.astype(np.float64)
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) / 2.0
    cd = (temp - x3 * x3 - y3 * y3) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-8:
        return None
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    radius = float(np.hypot(cx - x1, cy - y1))
    if not np.isfinite(radius) or radius <= 0:
        return None
    return float(cx), float(cy), radius


def least_squares_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    if len(points) < 3:
        return None
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_sq = c + cx * cx + cy * cy
    if radius_sq <= 0 or not np.isfinite(radius_sq):
        return None
    return float(cx), float(cy), float(np.sqrt(radius_sq))


def adjusted_circle_confidence(
    confidence: float,
    *,
    residual_median_px: float,
    limb_support_fraction: float,
    config: DetectionConfig,
) -> float:
    median_quality = config.distorted_limb_median_px / max(residual_median_px, 1e-9)
    support_quality = limb_support_fraction / max(config.distorted_limb_support_fraction, 1e-9)
    quality = min(1.0, median_quality, support_quality)
    return float(max(0.0, min(confidence, quality)))


def robust_circle_fit(
    points: np.ndarray,
    config: DetectionConfig,
    expected_radius: float | None = None,
) -> tuple[float, float, float, float] | None:
    if len(points) < config.min_limb_points:
        return None
    if len(points) > config.max_edge_points:
        step = max(1, len(points) // config.max_edge_points)
        points = points[::step]

    rng = np.random.default_rng(config.seed)
    best: tuple[int, float, float, float, np.ndarray] | None = None
    radius_floor = 3.0
    radius_ceiling = max(np.ptp(points[:, 0]), np.ptp(points[:, 1])) * 3.0

    for _ in range(config.ransac_iterations):
        idx = rng.choice(len(points), 3, replace=False)
        circle = circle_from_three(points[idx])
        if circle is None:
            continue
        cx, cy, radius = circle
        if radius < radius_floor or radius > radius_ceiling:
            continue
        if expected_radius is not None:
            radius_error = abs(radius - expected_radius) / max(expected_radius, 1.0)
            if radius_error > 0.35:
                continue
        tolerance = max(2.0, radius * config.ransac_tolerance_fraction)
        distances = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        inliers = np.abs(distances - radius) <= tolerance
        score = int(np.count_nonzero(inliers))
        if best is None or score > best[0]:
            best = (score, cx, cy, radius, inliers)

    if best is None:
        circle = least_squares_circle(points)
        if circle is None:
            return None
        cx, cy, radius = circle
        confidence = min(0.5, len(points) / max(config.min_limb_points * 4, 1))
        return cx, cy, radius, confidence

    score, cx, cy, radius, inliers = best
    inlier_points = points[inliers]
    refined = least_squares_circle(inlier_points)
    if refined is not None:
        cx, cy, radius = refined
    confidence = min(1.0, score / max(len(points) * 0.35, 1.0))
    return cx, cy, radius, float(confidence)


def contour_points(component_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(np.float32)


def detect_image(
    image: np.ndarray,
    filename: str,
    config: DetectionConfig | None = None,
    expected_radius: float | None = None,
) -> FrameDetection:
    config = config or DetectionConfig()
    height, width = image.shape[:2]
    work_image, scale = downsample_for_detection(image, config.work_max_dim)
    normalized = normalize_luminance(luminance(work_image), config)
    mask = make_mask(normalized, config)
    component = largest_plausible_component(mask, config)
    if component is None:
        return FrameDetection(filename=filename, width=width, height=height, status="failed")

    points = contour_points(component)
    if len(points) < config.min_limb_points:
        return FrameDetection(
            filename=filename,
            width=width,
            height=height,
            status="failed",
            flags=["insufficient_limb"],
        )

    scaled_expected = expected_radius * scale if expected_radius is not None else None
    fit = robust_circle_fit(points, config, expected_radius=scaled_expected)
    if fit is None:
        return FrameDetection(
            filename=filename,
            width=width,
            height=height,
            status="failed",
            flags=["insufficient_limb"],
        )

    cx, cy, radius, confidence = fit
    distances = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    residuals = np.abs(distances - radius)
    tolerance = max(2.0, radius * config.ransac_tolerance_fraction)
    limb_support_fraction = float(np.mean(residuals <= tolerance))
    residual_median_px = float(np.median(residuals) / scale)
    residual_p90_px = float(np.percentile(residuals, 90) / scale)
    confidence = adjusted_circle_confidence(
        confidence,
        residual_median_px=residual_median_px,
        limb_support_fraction=limb_support_fraction,
        config=config,
    )

    cx /= scale
    cy /= scale
    radius /= scale

    ys, xs = np.nonzero(component)
    flags: list[str] = []
    edge_margin = max(2, int(round(4 * scale)))
    if xs.min() <= edge_margin:
        flags.append("touches_left_edge")
    if xs.max() >= component.shape[1] - 1 - edge_margin:
        flags.append("touches_right_edge")
    if ys.min() <= edge_margin:
        flags.append("touches_top_edge")
    if ys.max() >= component.shape[0] - 1 - edge_margin:
        flags.append("touches_bottom_edge")
    if (
        residual_median_px >= config.distorted_limb_median_px
        or limb_support_fraction <= config.distorted_limb_support_fraction
    ):
        flags.append("distorted_limb_suspected")

    edge_flags = {
        "touches_left_edge",
        "touches_right_edge",
        "touches_top_edge",
        "touches_bottom_edge",
    }
    clipped = any(flag in edge_flags for flag in flags)
    if confidence < 0.35:
        status = "low_confidence"
    elif clipped:
        status = "clipped"
    else:
        status = "ok"

    return FrameDetection(
        filename=filename,
        width=width,
        height=height,
        center_x=float(cx),
        center_y=float(cy),
        radius=float(radius),
        confidence=float(confidence),
        status=status,
        flags=flags,
        limb_support_fraction=limb_support_fraction,
        circle_residual_median_px=residual_median_px,
        circle_residual_p90_px=residual_p90_px,
    )


def detect_file(
    path: str | Path,
    config: DetectionConfig | None = None,
    expected_radius: float | None = None,
) -> FrameDetection:
    image = read_exr(path)
    return detect_image(image, Path(path).name, config, expected_radius=expected_radius)
