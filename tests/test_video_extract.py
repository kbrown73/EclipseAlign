from __future__ import annotations

import numpy as np
import av
import OpenImageIO as oiio

from eclipse_align.cli import (
    exr_half_for_video_frame,
    extract_video_frames,
    normalize_video_inputs,
    srgb_to_linear,
    video_frame_to_exr_float,
    video_frame_filename,
)
from eclipse_align.exr_io import read_exr


def write_test_mp4(path, *, frame_count: int = 3) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=2)
    stream.width = 4
    stream.height = 3
    stream.pix_fmt = "yuv420p"
    try:
        for index in range(frame_count):
            image = np.full((3, 4, 3), index * 60, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def exr_pixel_format(path) -> str:
    inp = oiio.ImageInput.open(str(path))
    assert inp is not None
    try:
        return str(inp.spec().format)
    finally:
        inp.close()


def test_video_frame_filename_uses_configurable_digits():
    assert video_frame_filename(1) == "frame_000001.exr"
    assert video_frame_filename(42, digits=4) == "frame_0042.exr"


def test_exr_half_for_video_frame_auto_preserves_uint16_precision():
    assert exr_half_for_video_frame(np.zeros((2, 2, 3), dtype=np.uint8), "auto")
    assert not exr_half_for_video_frame(np.zeros((2, 2, 3), dtype=np.uint16), "auto")
    assert exr_half_for_video_frame(np.zeros((2, 2, 3), dtype=np.uint16), "half")
    assert not exr_half_for_video_frame(np.zeros((2, 2, 3), dtype=np.uint8), "float")


def test_video_frame_to_exr_float_scales_integer_frames():
    frame = np.array([0, 128, 255], dtype=np.uint8)

    converted = video_frame_to_exr_float(frame, transfer="none")

    np.testing.assert_allclose(converted, np.array([0.0, 128.0 / 255.0, 1.0], dtype=np.float32))


def test_video_frame_to_exr_float_converts_srgb_to_linear_by_default():
    frame = np.array([0, 128, 255], dtype=np.uint8)

    converted = video_frame_to_exr_float(frame)

    np.testing.assert_allclose(converted, srgb_to_linear(frame.astype(np.float32) / 255.0), rtol=1e-6)
    assert converted[1] < 128.0 / 255.0


def test_extract_video_frames_writes_exr_sequence(tmp_path):
    video_path = tmp_path / "clip.mp4"
    output_dir = tmp_path / "frames"
    write_test_mp4(video_path, frame_count=3)

    count = extract_video_frames(video_path, output_dir, digits=4)

    outputs = sorted(output_dir.glob("*.exr"))
    assert count == 3
    assert [path.name for path in outputs] == [
        "frame_0001.exr",
        "frame_0002.exr",
        "frame_0003.exr",
    ]
    assert exr_pixel_format(outputs[0]) == "half"
    frame = read_exr(outputs[0])
    assert frame.shape == (3, 4, 3)
    assert frame.dtype == np.float32
    assert float(frame.max()) <= 1.0


def test_extract_video_frames_concatenates_multiple_inputs(tmp_path):
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    output_dir = tmp_path / "frames"
    write_test_mp4(first_video, frame_count=2)
    write_test_mp4(second_video, frame_count=3)

    count = extract_video_frames([first_video, second_video], output_dir, digits=4)

    outputs = sorted(output_dir.glob("*.exr"))
    assert count == 5
    assert [path.name for path in outputs] == [
        "frame_0001.exr",
        "frame_0002.exr",
        "frame_0003.exr",
        "frame_0004.exr",
        "frame_0005.exr",
    ]


def test_normalize_video_inputs_preserves_repeat_and_argument_order():
    inputs = normalize_video_inputs([["first.mp4"], ["second.mp4", "third.mp4"]])

    assert [path.as_posix() for path in inputs] == ["first.mp4", "second.mp4", "third.mp4"]


def test_normalize_video_inputs_splits_quoted_multi_path_value():
    inputs = normalize_video_inputs([['"first video.mp4" second.mp4']])

    assert [path.as_posix() for path in inputs] == ["first video.mp4", "second.mp4"]
