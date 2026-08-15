from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from .detect import DetectionConfig, downsample_for_detection, luminance, normalize_luminance, robust_circle_fit
from .exr_io import read_exr
from .models import FrameDetection


@dataclass
class DustConfig:
    work_max_dim: int = 1400
    disk_radius_fraction: float = 1.03
    min_normalized_luminance: float = 0.04
    blur_sigma_radius_fraction: float = 0.045
    min_blur_sigma: float = 3.0
    min_deficit: float = 0.030
    min_support_frames: int = 6
    min_hit_frames: int = 3
    min_hit_fraction: float = 0.10
    min_component_area: int = 4
    max_component_area_fraction: float = 0.01
    min_component_extent: float = 0.18
    frame_edge_margin_px: int = 32
    edge_veto_percentile: float = 98.5
    edge_veto_dilation_radius_fraction: float = 0.012
    edge_veto_min_component_area_fraction: float = 0.003
    edge_veto_min_component_aspect_ratio: float = 5.0
    max_candidate_component_area_fraction: float = 0.02
    max_candidate_component_aspect_ratio: float = 6.0
    min_candidate_component_extent: float = 0.18
    moon_limb_min_component_area: int = 64
    moon_limb_min_points: int = 96
    moon_limb_radius_tolerance_fraction: float = 0.25
    moon_limb_min_support_fraction: float = 0.48
    moon_limb_min_arc_span_degrees: float = 45.0
    moon_limb_max_center_distance_radius_fraction: float = 2.5
    moon_limb_veto_band_radius_fraction: float = 0.07
    solar_limb_veto_band_radius_fraction: float = 0.035
    solar_limb_min_component_band_fraction: float = 0.45
    moon_shadow_max_normalized_luminance: float = 0.16
    moon_shadow_min_component_area: int = 32
    moon_shadow_min_component_area_fraction: float = 0.0012
    moon_shadow_boundary_band_radius_fraction: float = 0.035
    moon_shadow_min_component_boundary_fraction: float = 0.45


@dataclass
class DustComponent:
    component_id: int
    center_x: float
    center_y: float
    radius_px: float
    area_px: float
    mean_score: float
    max_score: float
    median_support_frames: float
    median_hit_frames: float


@dataclass
class LimbCircleFit:
    center_x: float
    center_y: float
    radius: float
    support_fraction: float
    arc_span_degrees: float


@dataclass
class LimbEllipseFit:
    center_x: float
    center_y: float
    major_radius: float
    minor_radius: float
    angle_deg: float


@dataclass
class DustFrameDiagnostics:
    sun_circle: LimbCircleFit
    sun_ellipse: LimbEllipseFit | None
    moon_circle: LimbCircleFit | None


@dataclass
class DustDetectionResult:
    width: int
    height: int
    map_width: int
    map_height: int
    scale: float
    frames_used: int
    score_map: np.ndarray
    support_map: np.ndarray
    hit_map: np.ndarray
    mask: np.ndarray
    components: list[DustComponent]


def analyze_dust_inputs(
    inputs: list[Path],
    detections: list[FrameDetection],
    output_dir: Path,
    *,
    config: DustConfig | None = None,
    preview_max_dim: int = 1600,
) -> DustDetectionResult:
    config = config or DustConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    def items() -> Iterable[tuple[str, np.ndarray, FrameDetection]]:
        for path, detection in zip(inputs, detections):
            yield path.name, read_exr(path), detection

    result = analyze_dust_arrays(
        items(),
        config=config,
        candidate_dir=candidate_dir,
        preview_max_dim=preview_max_dim,
        total=len(inputs),
    )
    write_dust_diagnostics(output_dir, result)
    return result


def analyze_dust_arrays(
    items: Iterable[tuple[str, np.ndarray, FrameDetection]],
    *,
    config: DustConfig | None = None,
    candidate_dir: Path | None = None,
    preview_max_dim: int = 1600,
    total: int | None = None,
) -> DustDetectionResult:
    config = config or DustConfig()
    support_map: np.ndarray | None = None
    hit_map: np.ndarray | None = None
    source_width = 0
    source_height = 0
    scale = 1.0
    frames_used = 0

    for filename, image, detection in tqdm(items, total=total, desc="dust"):
        if source_width == 0 or source_height == 0:
            source_height, source_width = image.shape[:2]
        candidate = dust_candidate_masks(image, detection, config)
        if candidate is None:
            continue
        candidate_mask, support_mask, work_image, frame_scale, frame_diagnostics = candidate
        if support_map is None:
            map_height, map_width = candidate_mask.shape
            support_map = np.zeros((map_height, map_width), dtype=np.float32)
            hit_map = np.zeros((map_height, map_width), dtype=np.float32)
            scale = frame_scale
        elif candidate_mask.shape != support_map.shape:
            continue

        support_map += support_mask.astype(np.float32)
        hit_map += candidate_mask.astype(np.float32)
        frames_used += 1

        if candidate_dir is not None:
            write_candidate_preview(
                candidate_dir / f"{Path(filename).stem}.png",
                work_image,
                candidate_mask,
                support_mask,
                frame_diagnostics,
                preview_max_dim=preview_max_dim,
            )

    if support_map is None or hit_map is None:
        empty = np.zeros((1, 1), dtype=np.float32)
        return DustDetectionResult(
            width=source_width,
            height=source_height,
            map_width=1,
            map_height=1,
            scale=scale,
            frames_used=0,
            score_map=empty,
            support_map=empty,
            hit_map=empty,
            mask=empty.astype(np.uint8),
            components=[],
        )

    result = build_dust_result(
        support_map,
        hit_map,
        width=source_width,
        height=source_height,
        scale=scale,
        frames_used=frames_used,
        config=config,
    )
    return result


def dust_candidate_masks(
    image: np.ndarray,
    detection: FrameDetection,
    config: DustConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, DustFrameDiagnostics] | None:
    if detection.radius is None or detection.raw_center_x is None or detection.raw_center_y is None:
        return None
    if detection.status not in {"ok", "clipped", "low_confidence", "estimated"}:
        return None

    work_image, scale = downsample_for_detection(image, config.work_max_dim)
    luma = luminance(work_image)
    finite = np.isfinite(luma)
    if not np.any(finite):
        return None
    luma = np.where(finite, luma, 0.0).astype(np.float32)
    normalized_luma = normalize_luminance(luma, DetectionConfig())

    radius = float(detection.radius) * scale
    center_x = float(detection.raw_center_x) * scale
    center_y = float(detection.raw_center_y) * scale
    disk_mask = circular_mask(
        luma.shape,
        center_x=center_x,
        center_y=center_y,
        radius=radius * config.disk_radius_fraction,
    )
    if not np.any(disk_mask):
        return None

    sigma = max(config.min_blur_sigma, radius * config.blur_sigma_radius_fraction)
    smooth = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized_reference = normalize_luminance(smooth, DetectionConfig())
    reference = np.abs(smooth) + max(float(np.percentile(np.abs(smooth[disk_mask]), 85)) * 1e-4, 1e-6)
    deficit = np.maximum((smooth - luma) / reference, 0.0)

    frame_margin = min(
        int(round(config.frame_edge_margin_px * scale)),
        int(round(min(luma.shape) * 0.025)),
    )
    frame_mask = inner_frame_mask(luma.shape, frame_margin)
    edge_veto = strong_edge_veto(luma, disk_mask, radius, config)
    support_mask = disk_mask & frame_mask & (normalized_reference >= config.min_normalized_luminance)
    candidate_mask = support_mask & (deficit >= config.min_deficit)
    candidate_mask &= ~edge_veto
    candidate_mask = clean_binary_mask(candidate_mask)
    candidate_mask = filter_candidate_components(candidate_mask, config)
    moon_fit = fit_moon_limb_candidate(
        candidate_mask,
        sun_center_x=center_x,
        sun_center_y=center_y,
        sun_radius=radius,
        config=config,
    )
    if moon_fit is not None:
        candidate_mask &= ~limb_circle_band_mask(candidate_mask.shape, moon_fit, config.moon_limb_veto_band_radius_fraction)
    solar_limb_veto = solar_limb_candidate_veto(
        candidate_mask,
        sun_center_x=center_x,
        sun_center_y=center_y,
        sun_radius=radius,
        config=config,
    )
    candidate_mask &= ~solar_limb_veto
    moon_shadow_veto = moon_shadow_boundary_veto(
        candidate_mask,
        normalized_luma=normalized_luma,
        disk_mask=disk_mask,
        sun_radius=radius,
        config=config,
    )
    candidate_mask &= ~moon_shadow_veto
    candidate_mask = filter_candidate_components(candidate_mask, config)
    frame_diagnostics = DustFrameDiagnostics(
        sun_circle=LimbCircleFit(
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            support_fraction=1.0,
            arc_span_degrees=360.0,
        ),
        sun_ellipse=scaled_sun_ellipse(detection, scale),
        moon_circle=moon_fit,
    )
    return candidate_mask, support_mask, work_image, scale, frame_diagnostics


def build_dust_result(
    support_map: np.ndarray,
    hit_map: np.ndarray,
    *,
    width: int,
    height: int,
    scale: float,
    frames_used: int,
    config: DustConfig,
) -> DustDetectionResult:
    score_map = np.zeros_like(hit_map, dtype=np.float32)
    valid = support_map > 0
    score_map[valid] = hit_map[valid] / support_map[valid]

    mask = (
        (support_map >= config.min_support_frames)
        & (hit_map >= config.min_hit_frames)
        & (score_map >= config.min_hit_fraction)
    )
    mask = clean_binary_mask(mask)
    components = dust_components(mask, score_map, support_map, hit_map, scale, config)
    component_mask = filtered_component_mask(mask, config)
    return DustDetectionResult(
        width=width,
        height=height,
        map_width=score_map.shape[1],
        map_height=score_map.shape[0],
        scale=scale,
        frames_used=frames_used,
        score_map=score_map,
        support_map=support_map,
        hit_map=hit_map,
        mask=component_mask,
        components=components,
    )


def dust_components(
    mask: np.ndarray,
    score_map: np.ndarray,
    support_map: np.ndarray,
    hit_map: np.ndarray,
    scale: float,
    config: DustConfig,
) -> list[DustComponent]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    max_area = mask.shape[0] * mask.shape[1] * config.max_component_area_fraction
    components = []
    component_id = 1
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        extent = component_extent(area, width, height)
        if area < config.min_component_area or area > max_area or extent < config.min_component_extent:
            continue
        region = labels == label
        center_x, center_y = centroids[label]
        area_source = area / max(scale * scale, 1e-9)
        radius_px = float(np.sqrt(area_source / np.pi))
        components.append(
            DustComponent(
                component_id=component_id,
                center_x=float(center_x / scale),
                center_y=float(center_y / scale),
                radius_px=radius_px,
                area_px=float(area_source),
                mean_score=float(np.mean(score_map[region])),
                max_score=float(np.max(score_map[region])),
                median_support_frames=float(np.median(support_map[region])),
                median_hit_frames=float(np.median(hit_map[region])),
            )
        )
        component_id += 1
    components.sort(key=lambda item: item.max_score, reverse=True)
    for idx, component in enumerate(components, start=1):
        component.component_id = idx
    return components


def filtered_component_mask(mask: np.ndarray, config: DustConfig) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    output = np.zeros(mask.shape, dtype=np.uint8)
    max_area = mask.shape[0] * mask.shape[1] * config.max_component_area_fraction
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        extent = component_extent(area, width, height)
        if config.min_component_area <= area <= max_area and extent >= config.min_component_extent:
            output[labels == label] = 255
    return output


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    output = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=1)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=1)
    return output.astype(bool)


def filter_candidate_components(mask: np.ndarray, config: DustConfig) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    output = np.zeros(mask.shape, dtype=bool)
    max_area = mask.shape[0] * mask.shape[1] * config.max_candidate_component_area_fraction
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area > max_area:
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > config.max_candidate_component_aspect_ratio:
            continue
        extent = component_extent(area, width, height)
        if extent < config.min_candidate_component_extent:
            continue
        output[labels == label] = True
    return output


def component_extent(area: int, width: int, height: int) -> float:
    return area / max(width * height, 1)


def moon_limb_candidate_veto(
    candidate_mask: np.ndarray,
    *,
    sun_center_x: float,
    sun_center_y: float,
    sun_radius: float,
    config: DustConfig,
) -> np.ndarray:
    fit = fit_moon_limb_candidate(
        candidate_mask,
        sun_center_x=sun_center_x,
        sun_center_y=sun_center_y,
        sun_radius=sun_radius,
        config=config,
    )
    if fit is None:
        return np.zeros(candidate_mask.shape, dtype=bool)
    return limb_circle_band_mask(candidate_mask.shape, fit, config.moon_limb_veto_band_radius_fraction)


def fit_moon_limb_candidate(
    candidate_mask: np.ndarray,
    *,
    sun_center_x: float,
    sun_center_y: float,
    sun_radius: float,
    config: DustConfig,
) -> LimbCircleFit | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask.astype(np.uint8), connectivity=8)
    point_sets = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.moon_limb_min_component_area:
            continue
        ys, xs = np.nonzero(labels == label)
        point_sets.append(np.column_stack([xs, ys]).astype(np.float32))

    if not point_sets:
        return None
    points = np.vstack(point_sets)
    if len(points) < config.moon_limb_min_points:
        return None

    fit_config = DetectionConfig(
        work_max_dim=config.work_max_dim,
        min_limb_points=config.moon_limb_min_points,
        ransac_iterations=600,
    )
    fit = robust_circle_fit(points, fit_config, expected_radius=sun_radius)
    if fit is None:
        return None

    moon_center_x, moon_center_y, moon_radius, _ = fit
    if not all(np.isfinite(value) for value in (moon_center_x, moon_center_y, moon_radius)):
        return None
    radius_error = abs(moon_radius - sun_radius) / max(sun_radius, 1.0)
    if radius_error > config.moon_limb_radius_tolerance_fraction:
        return None
    center_distance = float(np.hypot(moon_center_x - sun_center_x, moon_center_y - sun_center_y))
    if center_distance > sun_radius * config.moon_limb_max_center_distance_radius_fraction:
        return None

    residuals = np.abs(np.hypot(points[:, 0] - moon_center_x, points[:, 1] - moon_center_y) - moon_radius)
    support_tolerance = max(2.0, moon_radius * config.moon_limb_veto_band_radius_fraction * 0.5)
    inliers = residuals <= support_tolerance
    support_fraction = float(np.mean(inliers))
    if support_fraction < config.moon_limb_min_support_fraction:
        return None
    arc_span_degrees = angular_span_degrees(points[inliers], moon_center_x, moon_center_y)
    if arc_span_degrees < config.moon_limb_min_arc_span_degrees:
        return None

    return LimbCircleFit(
        center_x=float(moon_center_x),
        center_y=float(moon_center_y),
        radius=float(moon_radius),
        support_fraction=support_fraction,
        arc_span_degrees=arc_span_degrees,
    )


def limb_circle_band_mask(
    shape: tuple[int, int],
    fit: LimbCircleFit,
    band_radius_fraction: float,
) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    band_width = max(3.0, fit.radius * band_radius_fraction)
    distance = np.hypot(xx - fit.center_x, yy - fit.center_y)
    return np.abs(distance - fit.radius) <= band_width


def angular_span_degrees(points: np.ndarray, center_x: float, center_y: float) -> float:
    if len(points) == 0:
        return 0.0
    angles = np.sort((np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x) + 2.0 * np.pi) % (2.0 * np.pi))
    gaps = np.diff(np.concatenate([angles, [angles[0] + 2.0 * np.pi]]))
    span = 2.0 * np.pi - float(np.max(gaps))
    return float(np.degrees(span))


def scaled_sun_ellipse(detection: FrameDetection, scale: float) -> LimbEllipseFit | None:
    if (
        detection.ellipse_center_x is None
        or detection.ellipse_center_y is None
        or detection.ellipse_major_radius is None
        or detection.ellipse_minor_radius is None
        or detection.ellipse_angle_deg is None
    ):
        return None
    return LimbEllipseFit(
        center_x=float(detection.ellipse_center_x) * scale,
        center_y=float(detection.ellipse_center_y) * scale,
        major_radius=float(detection.ellipse_major_radius) * scale,
        minor_radius=float(detection.ellipse_minor_radius) * scale,
        angle_deg=float(detection.ellipse_angle_deg),
    )


def solar_limb_candidate_veto(
    candidate_mask: np.ndarray,
    *,
    sun_center_x: float,
    sun_center_y: float,
    sun_radius: float,
    config: DustConfig,
) -> np.ndarray:
    yy, xx = np.ogrid[: candidate_mask.shape[0], : candidate_mask.shape[1]]
    band_width = max(1.5, sun_radius * config.solar_limb_veto_band_radius_fraction)
    solar_limb_band = np.abs(np.hypot(xx - sun_center_x, yy - sun_center_y) - sun_radius) <= band_width

    count, labels, _, _ = cv2.connectedComponentsWithStats(candidate_mask.astype(np.uint8), connectivity=8)
    veto = np.zeros(candidate_mask.shape, dtype=bool)
    for label in range(1, count):
        region = labels == label
        if float(np.mean(solar_limb_band[region])) >= config.solar_limb_min_component_band_fraction:
            veto[region] = True
    return veto


def moon_shadow_boundary_veto(
    candidate_mask: np.ndarray,
    *,
    normalized_luma: np.ndarray,
    disk_mask: np.ndarray,
    sun_radius: float,
    config: DustConfig,
) -> np.ndarray:
    dark_mask = (
        disk_mask
        & np.isfinite(normalized_luma)
        & (normalized_luma <= config.moon_shadow_max_normalized_luminance)
    )
    if not np.any(dark_mask):
        return np.zeros(candidate_mask.shape, dtype=bool)

    clean_kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_OPEN, clean_kernel, iterations=1).astype(bool)
    dark_mask &= disk_mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros(candidate_mask.shape, dtype=bool)

    min_area = max(
        config.moon_shadow_min_component_area,
        int(round(np.pi * sun_radius * sun_radius * config.moon_shadow_min_component_area_fraction)),
    )
    boundary_radius = max(2, int(round(sun_radius * config.moon_shadow_boundary_band_radius_fraction)))
    boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * boundary_radius + 1, 2 * boundary_radius + 1),
    )
    boundary_band = np.zeros(candidate_mask.shape, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        dilated = cv2.dilate(component.astype(np.uint8), boundary_kernel, iterations=1).astype(bool)
        eroded = cv2.erode(component.astype(np.uint8), boundary_kernel, iterations=1).astype(bool)
        boundary_band |= dilated & ~eroded

    if not np.any(boundary_band):
        return np.zeros(candidate_mask.shape, dtype=bool)

    candidate_count, candidate_labels, _, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8),
        connectivity=8,
    )
    veto = np.zeros(candidate_mask.shape, dtype=bool)
    for label in range(1, candidate_count):
        region = candidate_labels == label
        if float(np.mean(boundary_band[region])) >= config.moon_shadow_min_component_boundary_fraction:
            veto[region] = True
    return veto


def strong_edge_veto(
    luma: np.ndarray,
    disk_mask: np.ndarray,
    radius: float,
    config: DustConfig,
) -> np.ndarray:
    normalized = normalize_luminance(luma, DetectionConfig())
    smoothed = cv2.GaussianBlur(normalized, (0, 0), sigmaX=1.0, sigmaY=1.0)
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(grad_x, grad_y)
    values = gradient[disk_mask]
    if values.size == 0:
        return np.zeros(luma.shape, dtype=bool)
    threshold = float(np.percentile(values, config.edge_veto_percentile))
    if threshold <= 0:
        return np.zeros(luma.shape, dtype=bool)
    edge = (gradient >= threshold) & disk_mask
    edge = filter_strong_edge_components(edge, config)
    dilation_radius = max(1, int(round(radius * config.edge_veto_dilation_radius_fraction)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1),
    )
    edge = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1).astype(bool)
    return edge


def filter_strong_edge_components(mask: np.ndarray, config: DustConfig) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    output = np.zeros(mask.shape, dtype=bool)
    min_area = mask.shape[0] * mask.shape[1] * config.edge_veto_min_component_area_fraction
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(width, height) / max(min(width, height), 1)
        if area >= min_area or aspect >= config.edge_veto_min_component_aspect_ratio:
            output[labels == label] = True
    return output


def inner_frame_mask(shape: tuple[int, int], margin: int) -> np.ndarray:
    height, width = shape
    mask = np.ones(shape, dtype=bool)
    if margin <= 0:
        return mask
    clipped_margin = min(margin, max(min(height, width) // 2 - 1, 0))
    if clipped_margin <= 0:
        return mask
    mask[:clipped_margin, :] = False
    mask[-clipped_margin:, :] = False
    mask[:, :clipped_margin] = False
    mask[:, -clipped_margin:] = False
    return mask


def circular_mask(
    shape: tuple[int, int],
    *,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return np.hypot(xx - center_x, yy - center_y) <= radius


def write_dust_diagnostics(output_dir: Path, result: DustDetectionResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_component_summary(output_dir / "dust_summary.csv", result)
    cv2.imwrite(str(output_dir / "dust_mask.png"), result.mask)
    cv2.imwrite(str(output_dir / "dust_score.png"), score_preview(result.score_map, result.support_map))
    cv2.imwrite(str(output_dir / "dust_support.png"), count_preview(result.support_map))
    cv2.imwrite(str(output_dir / "dust_hits.png"), count_preview(result.hit_map))


def write_component_summary(path: Path, result: DustDetectionResult) -> None:
    fieldnames = [
        "component_id",
        "center_x",
        "center_y",
        "radius_px",
        "area_px",
        "mean_score",
        "max_score",
        "median_support_frames",
        "median_hit_frames",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for component in result.components:
            writer.writerow(component.__dict__)


def write_candidate_preview(
    path: Path,
    image: np.ndarray,
    candidate_mask: np.ndarray,
    support_mask: np.ndarray,
    frame_diagnostics: DustFrameDiagnostics,
    *,
    preview_max_dim: int,
) -> None:
    preview = tone_preview(image)
    overlay = preview.copy()
    overlay[support_mask] = (0.65 * overlay[support_mask] + np.array([40, 80, 40])).astype(np.uint8)
    overlay[candidate_mask] = (40, 40, 255)
    preview = cv2.addWeighted(preview, 0.45, overlay, 0.55, 0)
    draw_limb_overlays(preview, frame_diagnostics)
    if preview_max_dim > 0:
        height, width = preview.shape[:2]
        largest = max(width, height)
        if largest > preview_max_dim:
            scale = preview_max_dim / largest
            preview = cv2.resize(
                preview,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), preview):
        raise RuntimeError(f"Could not write dust candidate preview: {path}")


def draw_limb_overlays(preview: np.ndarray, frame_diagnostics: DustFrameDiagnostics) -> None:
    sun_color = (0, 220, 255)
    moon_color = (255, 220, 0)
    if frame_diagnostics.sun_ellipse is not None:
        draw_limb_ellipse(preview, frame_diagnostics.sun_ellipse, sun_color)
    else:
        draw_limb_circle(preview, frame_diagnostics.sun_circle, sun_color)
    if frame_diagnostics.moon_circle is not None:
        draw_limb_circle(preview, frame_diagnostics.moon_circle, moon_color)


def draw_limb_circle(preview: np.ndarray, fit: LimbCircleFit, color: tuple[int, int, int]) -> None:
    cv2.circle(
        preview,
        (int(round(fit.center_x)), int(round(fit.center_y))),
        max(1, int(round(fit.radius))),
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def draw_limb_ellipse(preview: np.ndarray, fit: LimbEllipseFit, color: tuple[int, int, int]) -> None:
    cv2.ellipse(
        preview,
        (int(round(fit.center_x)), int(round(fit.center_y))),
        (max(1, int(round(fit.major_radius))), max(1, int(round(fit.minor_radius)))),
        fit.angle_deg,
        0,
        360,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def tone_preview(image: np.ndarray) -> np.ndarray:
    normalized = normalize_luminance(luminance(image), DetectionConfig())
    preview = np.clip(np.power(normalized, 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)


def score_preview(score_map: np.ndarray, support_map: np.ndarray) -> np.ndarray:
    scaled = np.clip(score_map * 255.0, 0, 255).astype(np.uint8)
    scaled[support_map <= 0] = 0
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)


def count_preview(count_map: np.ndarray) -> np.ndarray:
    maximum = float(np.max(count_map))
    if maximum <= 0:
        return np.zeros(count_map.shape, dtype=np.uint8)
    return np.clip(count_map / maximum * 255.0, 0, 255).astype(np.uint8)
