from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .exr_io import read_exr, write_exr
from .models import FrameDetection


_DUST_MASK_CACHE: dict[str, np.ndarray] = {}


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


def load_dust_mask(path: str | Path) -> np.ndarray:
    cache_key = str(path)
    cached = _DUST_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    mask = cv2.imread(cache_key, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read dust mask: {path}")
    mask = mask > 0
    _DUST_MASK_CACHE[cache_key] = mask
    return mask


def resize_dust_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if mask.shape == (height, width):
        return mask.astype(bool, copy=False)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def fitted_solar_disk_mask(shape: tuple[int, int], detection: FrameDetection) -> np.ndarray:
    center_x = detection.center_x if detection.center_x is not None else detection.raw_center_x
    center_y = detection.center_y if detection.center_y is not None else detection.raw_center_y
    radius = detection.radius if detection.radius is not None else detection.raw_radius
    mask = np.zeros(shape, dtype=bool)
    if (
        center_x is None
        or center_y is None
        or radius is None
        or not np.isfinite(center_x)
        or not np.isfinite(center_y)
        or not np.isfinite(radius)
        or radius <= 0
    ):
        return mask
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    mask[:, :] = np.hypot(xx - center_x, yy - center_y) <= radius
    return mask


def dust_correction_mask_for_frame(
    dust_mask: np.ndarray,
    shape: tuple[int, int],
    detection: FrameDetection,
) -> np.ndarray:
    return resize_dust_mask(dust_mask, shape) & fitted_solar_disk_mask(shape, detection)


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0 or not np.any(mask):
        return mask.astype(bool, copy=False)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius_px + 1, 2 * radius_px + 1),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def correct_dust_pixels(
    image: np.ndarray,
    dust_mask: np.ndarray,
    *,
    inpaint_radius: float = 5.0,
    dilation_px: int = 2,
    limit_mask: np.ndarray | None = None,
) -> np.ndarray:
    if inpaint_radius <= 0:
        raise ValueError("Dust correction radius must be greater than zero")
    if dilation_px < 0:
        raise ValueError("Dust mask dilation must be zero or greater")
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    output = np.asarray(image, dtype=np.float32).copy()
    mask = resize_dust_mask(dust_mask, output.shape[:2])
    limit = None if limit_mask is None else resize_dust_mask(limit_mask, output.shape[:2])
    if limit is not None:
        mask &= limit
    mask = dilate_mask(mask, dilation_px)
    if limit is not None:
        mask &= limit
    if not np.any(mask):
        return output

    inpaint_mask = mask.astype(np.uint8) * 255
    for channel in range(output.shape[2]):
        repaired = cv2.inpaint(
            output[:, :, channel],
            inpaint_mask,
            inpaint_radius,
            cv2.INPAINT_NS,
        )
        repaired = clamp_repaired_channel(
            repaired,
            output[:, :, channel],
            mask,
            neighborhood_radius=max(3, int(round(inpaint_radius * 3.0)) + dilation_px),
        )
        output[:, :, channel][mask] = repaired[mask]
    return output


def clamp_repaired_channel(
    repaired: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    *,
    neighborhood_radius: int,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return repaired
    output = repaired.copy()
    height, width = mask.shape
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        left = max(0, x - neighborhood_radius)
        top = max(0, y - neighborhood_radius)
        right = min(width, x + w + neighborhood_radius)
        bottom = min(height, y + h + neighborhood_radius)
        local_original = original[top:bottom, left:right]
        local_mask = mask[top:bottom, left:right]
        samples = local_original[(~local_mask) & np.isfinite(local_original)]
        if samples.size < 8:
            continue
        low = float(np.percentile(samples, 1.0))
        high = float(np.percentile(samples, 99.0))
        if high < low:
            continue
        region = labels == label
        output[region] = np.clip(output[region], low, high)
    return output


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


def ellipse_alpha_mask(
    height: int,
    width: int,
    *,
    center_x: float,
    center_y: float,
    major_radius: float | None,
    minor_radius: float | None,
    angle_deg: float | None,
    dtype=np.float32,
) -> np.ndarray:
    mask = np.zeros((height, width, 1), dtype=dtype)
    if (
        major_radius is None
        or minor_radius is None
        or angle_deg is None
        or not np.isfinite(major_radius)
        or not np.isfinite(minor_radius)
        or major_radius <= 0
        or minor_radius <= 0
    ):
        return mask
    yy, xx = np.ogrid[:height, :width]
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = xx - center_x
    dy = yy - center_y
    major_coord = dx * cos_a + dy * sin_a
    minor_coord = -dx * sin_a + dy * cos_a
    normalized = (major_coord / major_radius) ** 2 + (minor_coord / minor_radius) ** 2
    mask[:, :, 0] = (normalized <= 1.0).astype(dtype)
    return mask


def add_circle_alpha(
    image: np.ndarray,
    detection: FrameDetection,
    crop: CropRect | None = None,
    *,
    rotation_deg: float = 0.0,
) -> np.ndarray:
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
    if (
        detection.ellipse_major_radius is not None
        and detection.ellipse_minor_radius is not None
        and detection.ellipse_angle_deg is not None
    ):
        alpha = ellipse_alpha_mask(
            output_height,
            output_width,
            center_x=center_x,
            center_y=center_y,
            major_radius=detection.ellipse_major_radius,
            minor_radius=detection.ellipse_minor_radius,
            angle_deg=detection.ellipse_angle_deg + rotation_deg,
            dtype=output.dtype,
        )
    else:
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
    dust_mask_path: str | Path | None = None,
    dust_correction_radius: float = 5.0,
    dust_mask_dilation: int = 2,
) -> None:
    if detection.translation_x is None or detection.translation_y is None:
        raise ValueError(f"Frame has no translation: {detection.filename}")
    image = read_exr(input_path)
    if dust_mask_path is not None:
        solar_disk_mask = fitted_solar_disk_mask(image.shape[:2], detection)
        dust_mask = resize_dust_mask(load_dust_mask(dust_mask_path), image.shape[:2])
        image = correct_dust_pixels(
            image,
            dust_mask,
            inpaint_radius=dust_correction_radius,
            dilation_px=dust_mask_dilation,
            limit_mask=solar_disk_mask,
        )
    aligned = translate_image(image, detection.translation_x, detection.translation_y)
    if apply_rotation and detection.rotation_confidence >= 0.6:
        aligned = rotate_image_around_center(aligned, detection.rotation_deg)
    if crop is not None:
        aligned = crop_with_padding(aligned, crop)
    if add_alpha_circle:
        alpha_rotation = detection.rotation_deg if apply_rotation and detection.rotation_confidence >= 0.6 else 0.0
        aligned = add_circle_alpha(aligned, detection, crop, rotation_deg=alpha_rotation)
    write_exr(output_path, aligned)
