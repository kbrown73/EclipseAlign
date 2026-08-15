from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detect import DetectionConfig, luminance, normalize_luminance
from .exr_io import read_exr
from .models import FrameDetection


COLOR_FINAL_OK = (80, 220, 80)
COLOR_FINAL_ESTIMATED = (0, 200, 255)
COLOR_FINAL_FAILED = (0, 0, 255)
COLOR_FINAL_CLIPPED = (0, 180, 255)
COLOR_RAW = (255, 80, 40)
COLOR_DRIFT = (255, 120, 255)
COLOR_TEXT = (255, 255, 255)


def tone_map_preview(image: np.ndarray, config: DetectionConfig | None = None) -> np.ndarray:
    config = config or DetectionConfig()
    luma = luminance(image)
    normalized = normalize_luminance(luma, config)
    preview = np.clip(np.power(normalized, 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)


def scaled_point(x: float, y: float, scale: float) -> tuple[int, int]:
    return int(round(x * scale)), int(round(y * scale))


def status_color(detection: FrameDetection) -> tuple[int, int, int]:
    if detection.status in {"low_confidence", "estimated"}:
        return COLOR_FINAL_ESTIMATED
    if detection.status == "failed":
        return COLOR_FINAL_FAILED
    if detection.status == "clipped":
        return COLOR_FINAL_CLIPPED
    return COLOR_FINAL_OK


def draw_transparent_circle(
    preview: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.55,
    thickness: int = 2,
) -> None:
    overlay = preview.copy()
    cv2.circle(overlay, center, radius, color, thickness)
    cv2.addWeighted(overlay, alpha, preview, 1.0 - alpha, 0.0, preview)


def draw_raw_fit(preview: np.ndarray, detection: FrameDetection, scale: float) -> None:
    if detection.raw_center_x is None or detection.raw_center_y is None:
        return
    center = scaled_point(detection.raw_center_x, detection.raw_center_y, scale)
    radius = detection.raw_radius if detection.raw_radius is not None else detection.radius
    if radius is not None:
        draw_transparent_circle(
            preview,
            center,
            int(round(radius * scale)),
            COLOR_RAW,
            alpha=0.55,
            thickness=2,
        )
    cv2.drawMarker(preview, center, COLOR_RAW, markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)


def draw_final_fit(preview: np.ndarray, detection: FrameDetection, scale: float) -> None:
    if detection.center_x is None or detection.center_y is None:
        return
    center = scaled_point(detection.center_x, detection.center_y, scale)
    color = status_color(detection)
    if detection.radius is not None:
        cv2.circle(preview, center, int(round(detection.radius * scale)), color, 2)
    cv2.drawMarker(preview, center, color, markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)


def draw_raw_drift_arrow(preview: np.ndarray, detection: FrameDetection, scale: float) -> None:
    if (
        detection.center_x is None
        or detection.center_y is None
        or detection.raw_center_x is None
        or detection.raw_center_y is None
    ):
        return
    start = scaled_point(detection.center_x, detection.center_y, scale)
    end = scaled_point(detection.raw_center_x, detection.raw_center_y, scale)
    distance = np.hypot(end[0] - start[0], end[1] - start[1])
    if distance < 4:
        return
    cv2.arrowedLine(preview, start, end, COLOR_DRIFT, 2, tipLength=0.12)


def detection_label_lines(detection: FrameDetection) -> list[str]:
    line = f"{Path(detection.filename).name}  {detection.status}  conf={detection.confidence:.2f}"
    quality = []
    if detection.circle_residual_median_px is not None:
        quality.append(f"med={detection.circle_residual_median_px:.1f}px")
    if detection.limb_support_fraction is not None:
        quality.append(f"support={detection.limb_support_fraction:.2f}")
    if detection.raw_center_x is not None and detection.raw_center_y is not None and detection.has_center:
        drift = np.hypot(detection.raw_center_x - detection.center_x, detection.raw_center_y - detection.center_y)
        if drift >= 1.0:
            quality.append(f"raw-offset={drift:.1f}px")
    lines = [line]
    if quality:
        lines.append("  ".join(quality))
    if detection.flags:
        lines.append("flags=" + ",".join(detection.flags[:4]))
    return lines


def draw_text_lines(
    preview: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    *,
    font_scale: float = 0.58,
    line_height: int = 24,
) -> None:
    x, y = origin
    for index, line in enumerate(lines):
        cv2.putText(
            preview,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )


def draw_legend(preview: np.ndarray) -> None:
    items = [
        ("final ok", COLOR_FINAL_OK),
        ("estimated/low", COLOR_FINAL_ESTIMATED),
        ("clipped", COLOR_FINAL_CLIPPED),
        ("failed", COLOR_FINAL_FAILED),
        ("raw pre-refine", COLOR_RAW),
        ("final -> raw", COLOR_DRIFT),
    ]
    x = 16
    y = 110
    spacing = 24
    for index, (label, color) in enumerate(items):
        row_y = y + index * spacing
        cv2.line(preview, (x, row_y - 5), (x + 24, row_y - 5), color, 3)
        cv2.putText(
            preview,
            label,
            (x + 34, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )


def write_overlay_preview(
    input_path: str | Path,
    output_path: str | Path,
    detection: FrameDetection,
    config: DetectionConfig | None = None,
    *,
    max_dim: int = 1600,
) -> None:
    image = read_exr(input_path)
    preview = tone_map_preview(image, config)
    scale = 1.0
    if max_dim > 0:
        height, width = preview.shape[:2]
        largest = max(width, height)
        if largest > max_dim:
            scale = max_dim / largest
            preview = cv2.resize(
                preview,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
    draw_raw_fit(preview, detection, scale)
    draw_final_fit(preview, detection, scale)
    draw_raw_drift_arrow(preview, detection, scale)
    draw_text_lines(preview, detection_label_lines(detection), (16, 30))
    draw_legend(preview)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Could not write preview: {output_path}")
