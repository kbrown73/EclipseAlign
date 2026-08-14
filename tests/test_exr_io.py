from __future__ import annotations

import numpy as np
import OpenImageIO as oiio

from astroio.exr import read_exr as read_astroio_exr
from eclipse_align.exr_io import as_working_float_frame, read_exr, write_exr


def write_test_exr(path, image: np.ndarray, pixel_type) -> None:
    image = np.asarray(image)
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
        image = image[:, :, np.newaxis]
    else:
        height, width, channels = image.shape
    spec = oiio.ImageSpec(width, height, channels, pixel_type)
    out = oiio.ImageOutput.create(str(path))
    assert out is not None
    try:
        assert out.open(str(path), spec)
        assert out.write_image(image)
    finally:
        out.close()


def test_read_exr_uses_astroio_but_preserves_eclipse_align_working_shape(tmp_path):
    path = tmp_path / "mono.exr"
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    write_test_exr(path, source, oiio.HALF)

    astroio_frame = read_astroio_exr(path)
    working_frame = read_exr(path)

    assert astroio_frame.shape == (2, 3)
    assert astroio_frame.dtype == np.float16
    assert working_frame.shape == (2, 3, 1)
    assert working_frame.dtype == np.float32
    np.testing.assert_array_equal(working_frame[:, :, 0], source.astype(np.float16).astype(np.float32))


def test_as_working_float_frame_preserves_channel_images():
    image = np.ones((2, 3, 3), dtype=np.float16)

    working_frame = as_working_float_frame(image)

    assert working_frame.shape == (2, 3, 3)
    assert working_frame.dtype == np.float32


def test_write_exr_uses_astroio_and_preserves_half_default(tmp_path):
    path = tmp_path / "nested" / "frame.exr"
    image = np.arange(6, dtype=np.float64).reshape(2, 3)

    write_exr(path, image)

    astroio_frame = read_astroio_exr(path)
    working_frame = read_exr(path)
    assert astroio_frame.dtype == np.float16
    assert astroio_frame.shape == (2, 3)
    assert working_frame.dtype == np.float32
    assert working_frame.shape == (2, 3, 1)


def test_write_exr_can_write_float_output(tmp_path):
    path = tmp_path / "frame.exr"
    image = np.ones((2, 3, 3), dtype=np.float32)

    write_exr(path, image, half=False)

    astroio_frame = read_astroio_exr(path)
    assert astroio_frame.dtype == np.float32
    assert astroio_frame.shape == (2, 3, 3)
