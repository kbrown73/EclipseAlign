import numpy as np

from eclipse_align.models import FrameDetection
from eclipse_align.render import rotate_image_around_center
from eclipse_align.rotation import RotationConfig, assign_segments, detect_reframe_boundaries, initialize_rotation_metadata


def test_detect_reframe_boundaries_from_center_jumps():
    detections = [
        FrameDetection("a.exr", 100, 100, center_x=10, center_y=10),
        FrameDetection("b.exr", 100, 100, center_x=12, center_y=10),
        FrameDetection("c.exr", 100, 100, center_x=90, center_y=90),
    ]

    assert detect_reframe_boundaries(detections, threshold_px=50) == [2]


def test_detect_reframe_boundaries_merges_tiny_segments():
    detections = [
        FrameDetection("a.exr", 100, 100, center_x=0, center_y=0),
        FrameDetection("b.exr", 100, 100, center_x=100, center_y=0),
        FrameDetection("c.exr", 100, 100, center_x=200, center_y=0),
        FrameDetection("d.exr", 100, 100, center_x=202, center_y=0),
    ]

    assert detect_reframe_boundaries(detections, threshold_px=50, min_segment_length=2) == [2]


def test_assign_segments_increments_at_boundaries():
    detections = [FrameDetection(f"{idx}.exr", 100, 100) for idx in range(5)]

    assign_segments(detections, [2, 4])

    assert [d.segment_id for d in detections] == [0, 0, 1, 1, 2]


def test_rotate_image_around_center_preserves_shape():
    image = np.zeros((20, 20, 1), dtype=np.float32)
    image[5:15, 9:11, 0] = 1.0

    rotated = rotate_image_around_center(image, 10)

    assert rotated.shape == image.shape
    assert rotated.max() > 0


def test_initialize_rotation_metadata_resets_fields():
    detections = [
        FrameDetection(
            "a.exr",
            100,
            100,
            rotation_deg=12,
            rotation_confidence=1,
            rotation_source="registration",
        )
    ]

    initialize_rotation_metadata(detections)

    assert detections[0].rotation_deg == 0
    assert detections[0].rotation_confidence == 0
    assert detections[0].rotation_source == "none"


def test_rotation_config_defaults_are_conservative():
    config = RotationConfig()

    assert config.min_boundary_confidence >= 0.6
