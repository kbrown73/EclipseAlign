from __future__ import annotations

import glob
import re
from pathlib import Path


def numeric_sort_key(path: str | Path) -> tuple[str, int, str]:
    name = Path(path).name
    numbers = re.findall(r"\d+", name)
    if not numbers:
        return (name, -1, name)
    return (name[: name.find(numbers[-1])], int(numbers[-1]), name)


def discover_inputs(pattern_or_dir: str) -> list[Path]:
    path = Path(pattern_or_dir)
    if path.is_dir():
        files = list(path.glob("*.exr"))
    else:
        files = [Path(p) for p in glob.glob(pattern_or_dir)]
    return sorted(files, key=numeric_sort_key)
