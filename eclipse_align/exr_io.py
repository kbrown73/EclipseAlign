from __future__ import annotations

from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from astroio.exr import read_exr as read_astroio_exr
from astroio.exr import write_exr as write_astroio_exr


def read_exr(path: str | Path) -> np.ndarray:
    return as_working_float_frame(read_astroio_exr(path))


def as_working_float_frame(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.ndim != 3:
        raise ValueError(f"Expected 2D mono or 3D channel image, got shape {image.shape}")
    return np.asarray(image, dtype=np.float32)


def write_exr(path: str | Path, image: np.ndarray, *, half: bool = True) -> None:
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.ndim != 3:
        raise ValueError(f"Expected 2D mono or 3D channel image, got shape {image.shape}")
    write_astroio_exr(path, image.astype(np.float32, copy=False), half=half)


def image_size(path: str | Path) -> tuple[int, int, int]:
    path = str(path)
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(f"Could not open image: {path}")
    try:
        spec = inp.spec()
        return spec.width, spec.height, spec.nchannels
    finally:
        inp.close()
