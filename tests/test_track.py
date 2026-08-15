from eclipse_align.models import FrameDetection
from eclipse_align.track import refine_detections


def test_refine_detections_preserves_raw_centers_and_uses_detected_centers():
    detections = [
        FrameDetection(
            f"{idx}.exr",
            100,
            100,
            center_x=float(idx * 10 + (3 if idx == 2 else 0)),
            center_y=float(idx * 5),
            radius=10,
            confidence=1,
            status="ok",
        )
        for idx in range(5)
    ]

    refine_detections(detections)

    assert detections[2].raw_center_x == 23
    assert detections[2].raw_center_y == 10
    assert detections[2].raw_radius == 10
    assert detections[2].center_x == detections[2].raw_center_x
    assert detections[2].center_y == detections[2].raw_center_y
    assert detections[2].translation_x == 50 - detections[2].center_x
    assert detections[2].translation_y == 50 - detections[2].center_y


def test_refine_detections_interpolates_low_confidence_centers_and_preserves_raw():
    detections = [
        FrameDetection(
            f"{idx}.exr",
            100,
            100,
            center_x=90.0 if idx == 2 else float(idx * 10),
            center_y=80.0 if idx == 2 else float(idx * 5),
            radius=10,
            confidence=0.2 if idx == 2 else 1.0,
            status="low_confidence" if idx == 2 else "ok",
        )
        for idx in range(5)
    ]

    refine_detections(detections)

    assert detections[2].raw_center_x == 90.0
    assert detections[2].raw_center_y == 80.0
    assert detections[2].raw_radius == 10
    assert detections[2].center_x == 20.0
    assert detections[2].center_y == 10.0
    assert detections[2].status == "estimated"
    assert detections[2].translation_x == 30.0
    assert detections[2].translation_y == 40.0


def test_refine_detections_does_not_translate_unusable_low_confidence_without_neighbors():
    detections = [
        FrameDetection(
            "obstructed.exr",
            100,
            100,
            center_x=90.0,
            center_y=80.0,
            radius=10,
            confidence=0.2,
            status="low_confidence",
        )
    ]

    refine_detections(detections)

    assert detections[0].raw_center_x == 90.0
    assert detections[0].raw_center_y == 80.0
    assert detections[0].raw_radius == 10
    assert detections[0].center_x == 90.0
    assert detections[0].center_y == 80.0
    assert detections[0].translation_x is None
    assert detections[0].translation_y is None


def test_refine_detections_interpolates_distorted_limb_anchor():
    detections = [
        FrameDetection(
            f"{idx}.exr",
            100,
            100,
            center_x=90.0 if idx == 2 else float(idx * 10),
            center_y=80.0 if idx == 2 else float(idx * 5),
            radius=10,
            confidence=0.4 if idx == 2 else 1.0,
            status="ok",
            flags=["distorted_limb_suspected"] if idx == 2 else [],
        )
        for idx in range(5)
    ]

    refine_detections(detections)

    assert detections[2].raw_center_x == 90.0
    assert detections[2].raw_center_y == 80.0
    assert detections[2].raw_radius == 10
    assert detections[2].center_x == 20.0
    assert detections[2].center_y == 10.0
    assert detections[2].status == "estimated"
    assert "interpolated" in detections[2].flags


def test_refine_detections_prefers_clean_common_radius():
    detections = [
        FrameDetection("0.exr", 100, 100, center_x=0, center_y=0, radius=10, confidence=1, status="ok"),
        FrameDetection("1.exr", 100, 100, center_x=10, center_y=5, radius=10, confidence=1, status="ok"),
        FrameDetection(
            "2.exr",
            100,
            100,
            center_x=90,
            center_y=80,
            radius=40,
            confidence=0.4,
            status="ok",
            flags=["distorted_limb_suspected"],
        ),
        FrameDetection("3.exr", 100, 100, center_x=30, center_y=15, radius=10, confidence=1, status="ok"),
    ]

    common_radius = refine_detections(detections)

    assert common_radius == 10.0
    assert detections[2].raw_radius == 40.0
    assert detections[2].radius == 10.0
