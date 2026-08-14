from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from .detect import DetectionConfig, downsample_for_detection, luminance, normalize_luminance
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
    frame_edge_margin_px: int = 32
    edge_veto_percentile: float = 98.5
    edge_veto_dilation_radius_fraction: float = 0.012
    edge_veto_min_component_area_fraction: float = 0.003
    edge_veto_min_component_aspect_ratio: float = 5.0
    max_candidate_component_area_fraction: float = 0.02
    max_candidate_component_aspect_ratio: float = 6.0


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
        candidate_mask, support_mask, work_image, frame_scale = candidate
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
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
    return candidate_mask, support_mask, work_image, scale


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
        if area < config.min_component_area or area > max_area:
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
        if config.min_component_area <= area <= max_area:
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
        output[labels == label] = True
    return output


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
    *,
    preview_max_dim: int,
) -> None:
    preview = tone_preview(image)
    overlay = preview.copy()
    overlay[support_mask] = (0.65 * overlay[support_mask] + np.array([40, 80, 40])).astype(np.uint8)
    overlay[candidate_mask] = (40, 40, 255)
    preview = cv2.addWeighted(preview, 0.45, overlay, 0.55, 0)
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
