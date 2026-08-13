from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .exr_io import read_exr, write_exr
from .models import FrameDetection


@dataclass(frozen=True)
class CropRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


def compute_safe_crop(
    detections: list[FrameDetection],
    *,
    mode: str,
    square: bool = False,
    target_x: float | None = None,
    target_y: float | None = None,
    min_radius_margin: float = 1.05,
) -> CropRect | None:
    if mode == "none":
        return None
    if not detections:
        return None

    if mode == "safe-visible":
        selected = [d for d in detections if d.fully_visible_for_crop and d.confidence >= 0.25]
    elif mode == "safe":
        selected = [d for d in detections if d.translation_x is not None and d.translation_y is not None]
    else:
        raise ValueError(f"Unknown crop mode: {mode}")

    if not selected:
        return None

    width = selected[0].width
    height = selected[0].height
    left = 0.0
    top = 0.0
    right = float(width)
    bottom = float(height)
    for detection in selected:
        dx = float(detection.translation_x or 0.0)
        dy = float(detection.translation_y or 0.0)
        left = max(left, dx)
        top = max(top, dy)
        right = min(right, width + dx)
        bottom = min(bottom, height + dy)

    crop = CropRect(
        left=int(np.ceil(left)),
        top=int(np.ceil(top)),
        right=int(np.floor(right)),
        bottom=int(np.floor(bottom)),
    )
    if crop.width <= 0 or crop.height <= 0:
        return None
    if square:
        return centered_square_crop(
            crop,
            target_x=target_x if target_x is not None else width / 2.0,
            target_y=target_y if target_y is not None else height / 2.0,
            min_side=minimum_eclipse_crop_side(selected, min_radius_margin),
        )
    return crop


def compute_centered_square_crop(
    detections: list[FrameDetection],
    *,
    margin: int = 0,
    target_x: float | None = None,
    target_y: float | None = None,
    min_radius_margin: float = 1.05,
) -> CropRect | None:
    if margin < 0:
        raise ValueError("Margin must be zero or greater")
    safe_square = compute_safe_crop(
        detections,
        mode="safe-visible",
        square=True,
        target_x=target_x,
        target_y=target_y,
        min_radius_margin=min_radius_margin,
    )
    if safe_square is None:
        return None
    side = safe_square.width + 2 * margin
    return centered_fixed_square_crop(detections, side, target_x=target_x, target_y=target_y)


def minimum_eclipse_crop_side(detections: list[FrameDetection], margin: float) -> int:
    radii = [d.radius for d in detections if d.radius is not None and d.confidence >= 0.25]
    if not radii:
        return 0
    return int(np.ceil(2.0 * float(np.median(radii)) * margin))


def centered_square_crop(
    bounds: CropRect,
    *,
    target_x: float,
    target_y: float,
    min_side: int = 0,
) -> CropRect | None:
    max_half_side = min(
        target_x - bounds.left,
        bounds.right - target_x,
        target_y - bounds.top,
        bounds.bottom - target_y,
    )
    side = int(np.floor(2.0 * max_half_side))
    if side <= 0:
        return None
    if min_side and side < min_side:
        return None
    half = side / 2.0
    left = int(np.ceil(target_x - half))
    top = int(np.ceil(target_y - half))
    right = left + side
    bottom = top + side
    return CropRect(left=left, top=top, right=right, bottom=bottom)


def parse_manual_crop(value: str) -> CropRect:
    try:
        size, offset = value.split("+", 1)
        width_s, height_s = size.lower().split("x", 1)
        left_s, top_s = offset.split("+", 1)
        left = int(left_s)
        top = int(top_s)
        width = int(width_s)
        height = int(height_s)
    except ValueError as exc:
        raise ValueError("Manual crop must use WxH+X+Y, e.g. 1920x1080+100+50") from exc
    return CropRect(left=left, top=top, right=left + width, bottom=top + height)


def centered_fixed_square_crop(
    detections: list[FrameDetection],
    side: int,
    *,
    target_x: float | None = None,
    target_y: float | None = None,
) -> CropRect:
    if not detections:
        raise ValueError("Cannot compute fixed square crop without detections")
    if side <= 0:
        raise ValueError("Square crop side must be positive")
    width = detections[0].width
    height = detections[0].height
    tx = target_x if target_x is not None else width / 2.0
    ty = target_y if target_y is not None else height / 2.0
    half = side / 2.0
    left = int(round(tx - half))
    top = int(round(ty - half))
    right = left + side
    bottom = top + side
    return CropRect(left=left, top=top, right=right, bottom=bottom)


def crop_with_padding(image: np.ndarray, crop: CropRect) -> np.ndarray:
    height, width = image.shape[:2]
    src_left = max(crop.left, 0)
    src_top = max(crop.top, 0)
    src_right = min(crop.right, width)
    src_bottom = min(crop.bottom, height)

    if image.ndim == 2:
        output = np.zeros((crop.height, crop.width), dtype=image.dtype)
        if src_right > src_left and src_bottom > src_top:
            dst_left = src_left - crop.left
            dst_top = src_top - crop.top
            output[
                dst_top : dst_top + (src_bottom - src_top),
                dst_left : dst_left + (src_right - src_left),
            ] = image[src_top:src_bottom, src_left:src_right]
        return output

    output = np.zeros((crop.height, crop.width, image.shape[2]), dtype=image.dtype)
    if src_right > src_left and src_bottom > src_top:
        dst_left = src_left - crop.left
        dst_top = src_top - crop.top
        output[
            dst_top : dst_top + (src_bottom - src_top),
            dst_left : dst_left + (src_right - src_left),
        ] = image[src_top:src_bottom, src_left:src_right]
    return output


def rotate_image_around_center(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-9:
        return image
    single_channel = image.ndim == 3 and image.shape[2] == 1
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if single_channel and rotated.ndim == 2:
        rotated = rotated[:, :, np.newaxis]
    return rotated


def translate_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def circle_alpha_mask(
    height: int,
    width: int,
    *,
    center_x: float,
    center_y: float,
    radius: float | None,
    dtype=np.float32,
) -> np.ndarray:
    mask = np.zeros((height, width, 1), dtype=dtype)
    if radius is None or not np.isfinite(radius) or radius <= 0:
        return mask
    yy, xx = np.ogrid[:height, :width]
    distance = np.hypot(xx - center_x, yy - center_y)
    mask[:, :, 0] = (distance <= radius).astype(dtype)
    return mask


def add_circle_alpha(image: np.ndarray, detection: FrameDetection, crop: CropRect | None = None) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    output = np.asarray(image)
    output_height, output_width = output.shape[:2]
    source_center_x = detection.width / 2.0
    source_center_y = detection.height / 2.0
    if crop is None:
        center_x = source_center_x
        center_y = source_center_y
    else:
        center_x = source_center_x - crop.left
        center_y = source_center_y - crop.top
    alpha = circle_alpha_mask(
        output_height,
        output_width,
        center_x=center_x,
        center_y=center_y,
        radius=detection.radius,
        dtype=output.dtype,
    )
    if output.shape[2] >= 4:
        output = output.copy()
        output[:, :, 3] = alpha[:, :, 0]
        return output
    return np.concatenate([output, alpha], axis=2)


def render_frame(
    input_path: str | Path,
    output_path: str | Path,
    detection: FrameDetection,
    *,
    crop: CropRect | None = None,
    apply_rotation: bool = True,
    add_alpha_circle: bool = False,
) -> None:
    if detection.translation_x is None or detection.translation_y is None:
        raise ValueError(f"Frame has no translation: {detection.filename}")
    image = read_exr(input_path)
    aligned = translate_image(image, detection.translation_x, detection.translation_y)
    if apply_rotation and detection.rotation_confidence >= 0.6:
        aligned = rotate_image_around_center(aligned, detection.rotation_deg)
    if crop is not None:
        aligned = crop_with_padding(aligned, crop)
    if add_alpha_circle:
        aligned = add_circle_alpha(aligned, detection, crop)
    write_exr(output_path, aligned)
