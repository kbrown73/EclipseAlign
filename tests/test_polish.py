import cv2
import numpy as np

from eclipse_align.models import FrameDetection
from eclipse_align.polish import PolishConfig, estimate_polish_offsets


def shifted_crop(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = image.shape
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


def normalize(image: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(image))
    return (image / norm).astype(np.float32)


def test_estimate_polish_offsets_returns_correction_opposite_measured_shift():
    base = np.zeros((64, 64), dtype=np.float32)
    base[24:36, 28:40] = 1.0
    crops = [normalize(base), normalize(shifted_crop(base, 1.0, -1.0)), normalize(base)]
    detections = [
        FrameDetection(f"{idx}.exr", 64, 64, radius=20, segment_id=0)
        for idx in range(3)
    ]

    results = estimate_polish_offsets(crops, detections, PolishConfig(max_shift_px=2.0))

    assert results[1].source == "phase_correlation"
    assert abs(results[1].residual_dx + 1.0) < 0.1
    assert abs(results[1].residual_dy - 1.0) < 0.1


def test_estimate_polish_offsets_rejects_large_shift():
    base = np.zeros((64, 64), dtype=np.float32)
    base[24:36, 28:40] = 1.0
    crops = [normalize(base), normalize(shifted_crop(base, 6.0, 0.0)), normalize(base)]
    detections = [
        FrameDetection(f"{idx}.exr", 64, 64, radius=20, segment_id=0)
        for idx in range(3)
    ]

    results = estimate_polish_offsets(crops, detections, PolishConfig(max_shift_px=2.0))

    assert results[1].source == "rejected"
    assert results[1].residual_dx == 0
