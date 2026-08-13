from pathlib import Path

from eclipse_align.cli import build_render_payloads, reformatted_output_filename
from eclipse_align.models import FrameDetection


def test_reformatted_output_filename_starts_at_one():
    assert reformatted_output_filename(1) == "frame_0001.exr"
    assert reformatted_output_filename(42) == "frame_0042.exr"


def test_build_render_payloads_reformats_rendered_outputs_continuously():
    inputs = [Path("IMG_0001.exr"), Path("IMG_0002.exr"), Path("IMG_0003.exr")]
    detections = {
        "IMG_0001.exr": FrameDetection("IMG_0001.exr", 100, 100, translation_x=1.0, translation_y=1.0),
        "IMG_0002.exr": FrameDetection("IMG_0002.exr", 100, 100),
        "IMG_0003.exr": FrameDetection("IMG_0003.exr", 100, 100, translation_x=2.0, translation_y=2.0),
    }

    payloads, skipped, output_name_by_source = build_render_payloads(
        inputs,
        detections,
        Path("aligned"),
        None,
        True,
        reformat_output=True,
        add_alpha_circle=True,
    )

    assert skipped == 1
    assert [payload[1] for payload in payloads] == [
        "aligned/frame_0001.exr",
        "aligned/frame_0002.exr",
    ]
    assert [payload[5] for payload in payloads] == [True, True]
    assert output_name_by_source == {
        "IMG_0001.exr": "frame_0001.exr",
        "IMG_0003.exr": "frame_0002.exr",
    }
