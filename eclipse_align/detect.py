from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .exr_io import read_exr
from .models import FrameDetection


DEFAULT_THRESHOLD = 0.08


@dataclass
class DetectionConfig:
    work_max_dim: int = 1400
    threshold: float = DEFAULT_THRESHOLD
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
    max_ellipse_axis_ratio: float = 1.40
    max_ellipse_tilt_deg: float = 28.0
    min_ellipse_improvement: float = 0.20
    max_ellipse_median_residual_px: float = 18.0


@dataclass
class EllipseFit:
    center_x: float
    center_y: float
    major_radius: float
    minor_radius: float
    angle_deg: float
    confidence: float
    residual_median_px: float
    residual_p90_px: float
    support_fraction: float


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


def normalize_ellipse_angle(angle_deg: float) -> float:
    return float(((angle_deg + 90.0) % 180.0) - 90.0)


def ellipse_from_points(
    points: np.ndarray,
    config: DetectionConfig,
    expected_radius: float | None = None,
) -> tuple[float, float, float, float, float] | None:
    if len(points) < 5:
        return None
    try:
        (cx, cy), (width, height), angle = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
    except cv2.error:
        return None

    if not all(np.isfinite(value) for value in (cx, cy, width, height, angle)):
        return None
    if width <= 0 or height <= 0:
        return None

    if width >= height:
        major_radius = width / 2.0
        minor_radius = height / 2.0
        major_angle = angle
    else:
        major_radius = height / 2.0
        minor_radius = width / 2.0
        major_angle = angle + 90.0

    if major_radius <= 0 or minor_radius <= 0:
        return None
    axis_ratio = major_radius / max(minor_radius, 1e-9)
    if axis_ratio > config.max_ellipse_axis_ratio:
        return None
    if expected_radius is not None:
        radius_error = abs(major_radius - expected_radius) / max(expected_radius, 1.0)
        if radius_error > 0.40:
            return None

    major_angle = normalize_ellipse_angle(major_angle)
    if abs(major_angle) > config.max_ellipse_tilt_deg:
        return None

    return float(cx), float(cy), float(major_radius), float(minor_radius), major_angle


def ellipse_radial_residuals(points: np.ndarray, ellipse: tuple[float, float, float, float, float]) -> np.ndarray:
    cx, cy, major_radius, minor_radius, angle_deg = ellipse
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = points[:, 0].astype(np.float64) - cx
    dy = points[:, 1].astype(np.float64) - cy
    major_coord = dx * cos_a + dy * sin_a
    minor_coord = -dx * sin_a + dy * cos_a
    point_radius = np.hypot(major_coord, minor_coord)
    theta = np.arctan2(minor_coord, major_coord)
    boundary_radius = 1.0 / np.sqrt(
        (np.cos(theta) / major_radius) ** 2 + (np.sin(theta) / minor_radius) ** 2
    )
    return np.abs(point_radius - boundary_radius)


def robust_ellipse_fit(
    points: np.ndarray,
    config: DetectionConfig,
    expected_radius: float | None = None,
) -> EllipseFit | None:
    if len(points) < max(config.min_limb_points, 5):
        return None
    if len(points) > config.max_edge_points:
        step = max(1, len(points) // config.max_edge_points)
        points = points[::step]

    rng = np.random.default_rng(config.seed)
    best: tuple[int, tuple[float, float, float, float, float], np.ndarray] | None = None

    for _ in range(config.ransac_iterations):
        idx = rng.choice(len(points), 5, replace=False)
        ellipse = ellipse_from_points(points[idx], config, expected_radius=expected_radius)
        if ellipse is None:
            continue
        residuals = ellipse_radial_residuals(points, ellipse)
        tolerance = max(2.0, ellipse[2] * config.ransac_tolerance_fraction)
        inliers = residuals <= tolerance
        score = int(np.count_nonzero(inliers))
        if best is None or score > best[0]:
            best = (score, ellipse, inliers)

    if best is None:
        ellipse = ellipse_from_points(points, config, expected_radius=expected_radius)
        if ellipse is None:
            return None
        score = min(len(points), config.min_limb_points * 2)
    else:
        score, ellipse, inliers = best
        inlier_points = points[inliers]
        if len(inlier_points) >= 5:
            refined = ellipse_from_points(inlier_points, config, expected_radius=expected_radius)
            if refined is not None:
                ellipse = refined

    residuals = ellipse_radial_residuals(points, ellipse)
    tolerance = max(2.0, ellipse[2] * config.ransac_tolerance_fraction)
    support_fraction = float(np.mean(residuals <= tolerance))
    residual_median_px = float(np.median(residuals))
    residual_p90_px = float(np.percentile(residuals, 90))
    confidence = min(1.0, score / max(len(points) * 0.35, 1.0))
    return EllipseFit(
        center_x=ellipse[0],
        center_y=ellipse[1],
        major_radius=ellipse[2],
        minor_radius=ellipse[3],
        angle_deg=ellipse[4],
        confidence=float(confidence),
        residual_median_px=residual_median_px,
        residual_p90_px=residual_p90_px,
        support_fraction=support_fraction,
    )


def contour_points(component_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(np.float32)


def limb_fit_points(
    points: np.ndarray,
    mask_shape: tuple[int, int],
    edge_margin: int,
    config: DetectionConfig,
) -> np.ndarray:
    height, width = mask_shape
    interior = points[
        (points[:, 0] > edge_margin)
        & (points[:, 0] < width - 1 - edge_margin)
        & (points[:, 1] > edge_margin)
        & (points[:, 1] < height - 1 - edge_margin)
    ]
    if len(interior) >= config.min_limb_points:
        return interior
    return points


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

    edge_margin = max(2, int(round(4 * scale)))
    fit_points = limb_fit_points(points, component.shape, edge_margin, config)
    scaled_expected = expected_radius * scale if expected_radius is not None else None
    fit = robust_circle_fit(fit_points, config, expected_radius=scaled_expected)
    if fit is None:
        return FrameDetection(
            filename=filename,
            width=width,
            height=height,
            status="failed",
            flags=["insufficient_limb"],
        )

    cx, cy, radius, confidence = fit
    distances = np.hypot(fit_points[:, 0] - cx, fit_points[:, 1] - cy)
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


def fit_horizon_ellipse_image(
    image: np.ndarray,
    config: DetectionConfig | None = None,
    expected_radius: float | None = None,
) -> EllipseFit | None:
    config = config or DetectionConfig()
    work_image, scale = downsample_for_detection(image, config.work_max_dim)
    normalized = normalize_luminance(luminance(work_image), config)
    mask = make_mask(normalized, config)
    component = largest_plausible_component(mask, config)
    if component is None:
        return None

    points = contour_points(component)
    edge_margin = max(2, int(round(4 * scale)))
    fit_points = limb_fit_points(points, component.shape, edge_margin, config)
    scaled_expected = expected_radius * scale if expected_radius is not None else None
    fit = robust_ellipse_fit(fit_points, config, expected_radius=scaled_expected)
    if fit is None:
        return None

    return EllipseFit(
        center_x=fit.center_x / scale,
        center_y=fit.center_y / scale,
        major_radius=fit.major_radius / scale,
        minor_radius=fit.minor_radius / scale,
        angle_deg=fit.angle_deg,
        confidence=fit.confidence,
        residual_median_px=fit.residual_median_px / scale,
        residual_p90_px=fit.residual_p90_px / scale,
        support_fraction=fit.support_fraction,
    )


def fit_horizon_ellipse_file(
    path: str | Path,
    config: DetectionConfig | None = None,
    expected_radius: float | None = None,
) -> EllipseFit | None:
    image = read_exr(path)
    return fit_horizon_ellipse_image(image, config, expected_radius=expected_radius)


def detect_file(
    path: str | Path,
    config: DetectionConfig | None = None,
    expected_radius: float | None = None,
) -> FrameDetection:
    image = read_exr(path)
    return detect_image(image, Path(path).name, config, expected_radius=expected_radius)
