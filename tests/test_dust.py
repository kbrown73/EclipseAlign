import cv2
import numpy as np

from eclipse_align.dust import DustConfig, analyze_dust_arrays, dust_candidate_masks
from eclipse_align.models import FrameDetection


def synthetic_frame(
    center_x: float,
    *,
    width: int = 96,
    height: int = 72,
    radius: float = 26.0,
    dust_center: tuple[float, float] = (42.0, 36.0),
    dust_strength: float = 0.22,
) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    disk = np.hypot(xx - center_x, yy - height / 2.0) <= radius
    image = np.full((height, width, 3), 0.02, dtype=np.float32)
    image[disk] = 0.75

    dust = np.exp(-(((xx - dust_center[0]) ** 2 + (yy - dust_center[1]) ** 2) / (2.0 * 2.2**2)))
    image *= (1.0 - dust_strength * dust[:, :, np.newaxis]).astype(np.float32)

    solar_spot_x = center_x - 9.0
    solar_spot = np.exp(-(((xx - solar_spot_x) ** 2 + (yy - 34.0) ** 2) / (2.0 * 2.0**2)))
    image *= (1.0 - 0.25 * solar_spot[:, :, np.newaxis]).astype(np.float32)
    return image


def detection_for_frame(index: int, center_x: float) -> FrameDetection:
    return FrameDetection(
        filename=f"frame_{index:04d}.exr",
        width=96,
        height=72,
        center_x=center_x,
        center_y=36.0,
        raw_center_x=center_x,
        raw_center_y=36.0,
        radius=26.0,
        confidence=1.0,
        status="ok",
        translation_x=48.0 - center_x,
        translation_y=0.0,
    )


def synthetic_moon_edge_frame() -> np.ndarray:
    width = 96
    height = 72
    yy, xx = np.ogrid[:height, :width]
    solar_disk = np.hypot(xx - 48.0, yy - 36.0) <= 26.0
    moon_disk = np.hypot(xx - 62.0, yy - 36.0) <= 22.0
    image = np.full((height, width, 3), 0.02, dtype=np.float32)
    image[solar_disk] = 0.75
    image[solar_disk & moon_disk] = 0.04
    return image


def test_dust_candidate_mask_finds_local_dark_deficit():
    image = synthetic_frame(44.0)
    detection = detection_for_frame(0, 44.0)

    candidate_mask, support_mask, _, _ = dust_candidate_masks(
        image,
        detection,
        DustConfig(work_max_dim=96, min_deficit=0.035),
    )

    assert support_mask[36, 42]
    assert candidate_mask[36, 42]


def test_dust_candidate_mask_uses_near_limb_area():
    image = synthetic_frame(48.0, dust_center=(72.0, 36.0), dust_strength=0.50)
    detection = detection_for_frame(0, 48.0)

    candidate_mask, support_mask, _, _ = dust_candidate_masks(
        image,
        detection,
        DustConfig(work_max_dim=96),
    )

    assert support_mask[36, 72]
    assert candidate_mask[36, 72]


def test_dust_candidate_mask_rejects_overlapping_moon_edge():
    image = synthetic_moon_edge_frame()
    detection = detection_for_frame(0, 48.0)

    candidate_mask, _, _, _ = dust_candidate_masks(
        image,
        detection,
        DustConfig(work_max_dim=96),
    )

    assert int(candidate_mask.sum()) == 0


def test_dust_candidate_mask_rejects_frame_edge():
    image = synthetic_frame(12.0)
    detection = detection_for_frame(0, 12.0)

    candidate_mask, support_mask, _, _ = dust_candidate_masks(
        image,
        detection,
        DustConfig(work_max_dim=96),
    )

    assert not support_mask[36, 1]
    assert not candidate_mask[36, 1]


def test_analyze_dust_arrays_accumulates_sensor_fixed_spots():
    centers = np.linspace(36.0, 58.0, 18)
    items = [
        (f"frame_{idx:04d}.exr", synthetic_frame(float(center)), detection_for_frame(idx, float(center)))
        for idx, center in enumerate(centers)
    ]

    result = analyze_dust_arrays(
        items,
        config=DustConfig(
            work_max_dim=96,
            min_deficit=0.035,
            min_support_frames=8,
            min_hit_frames=5,
            min_hit_fraction=0.35,
            min_component_area=3,
        ),
    )

    assert result.components
    strongest = result.components[0]
    assert abs(strongest.center_x - 42.0) < 3.0
    assert abs(strongest.center_y - 36.0) < 3.0

    solar_frame_spot_hits = result.mask[:, 24:53]
    count, _, stats, centroids = cv2.connectedComponentsWithStats(solar_frame_spot_hits, connectivity=8)
    centroids_x = [float(centroids[label][0] + 24) for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= 3]
    assert not any(abs(x - 33.0) < 2.0 for x in centroids_x)
