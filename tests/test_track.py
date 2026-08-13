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
    assert detections[2].center_x == detections[2].raw_center_x
    assert detections[2].center_y == detections[2].raw_center_y
    assert detections[2].translation_x == 50 - detections[2].center_x
    assert detections[2].translation_y == 50 - detections[2].center_y
