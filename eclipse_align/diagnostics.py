from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detect import DetectionConfig, luminance, normalize_luminance
from .exr_io import read_exr
from .models import FrameDetection


def tone_map_preview(image: np.ndarray, config: DetectionConfig | None = None) -> np.ndarray:
    config = config or DetectionConfig()
    luma = luminance(image)
    normalized = normalize_luminance(luma, config)
    preview = np.clip(np.power(normalized, 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)


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
    if detection.center_x is not None and detection.center_y is not None:
        center = (int(round(detection.center_x * scale)), int(round(detection.center_y * scale)))
        color = (80, 220, 80)
        if detection.status in {"low_confidence", "estimated"}:
            color = (0, 200, 255)
        elif detection.status == "failed":
            color = (0, 0, 255)
        elif detection.status == "clipped":
            color = (0, 180, 255)
        if detection.radius is not None:
            cv2.circle(preview, center, int(round(detection.radius * scale)), color, 2)
        cv2.drawMarker(preview, center, color, markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
    label = f"{Path(detection.filename).name} {detection.status} conf={detection.confidence:.2f}"
    if detection.circle_residual_median_px is not None:
        label += f" med={detection.circle_residual_median_px:.1f}px"
    cv2.putText(preview, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3)
    cv2.putText(preview, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Could not write preview: {output_path}")
