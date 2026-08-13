from __future__ import annotations

from pathlib import Path

import numpy as np
import OpenImageIO as oiio


def read_exr(path: str | Path) -> np.ndarray:
    path = str(path)
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(f"Could not open EXR: {path}")
    try:
        image = inp.read_image(format=oiio.FLOAT)
    finally:
        inp.close()
    if image is None:
        raise RuntimeError(f"Could not read EXR pixels: {path}")
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    return np.asarray(image, dtype=np.float32)


def write_exr(path: str | Path, image: np.ndarray, *, half: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image = np.asarray(image)
    height, width, channels = image.shape
    pixel_type = oiio.HALF if half else oiio.FLOAT
    spec = oiio.ImageSpec(width, height, channels, pixel_type)
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"Could not create EXR output: {path}")
    try:
        if not out.open(str(path), spec):
            raise RuntimeError(f"Could not open EXR output: {path}: {out.geterror()}")
        if not out.write_image(image.astype(np.float32, copy=False)):
            raise RuntimeError(f"Could not write EXR output: {path}: {out.geterror()}")
    finally:
        out.close()


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
