from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FrameDetection:
    filename: str
    width: int
    height: int
    center_x: float | None = None
    center_y: float | None = None
    raw_center_x: float | None = None
    raw_center_y: float | None = None
    raw_radius: float | None = None
    radius: float | None = None
    confidence: float = 0.0
    status: str = "failed"
    flags: list[str] = field(default_factory=list)
    translation_x: float | None = None
    translation_y: float | None = None
    limb_support_fraction: float | None = None
    circle_residual_median_px: float | None = None
    circle_residual_p90_px: float | None = None
    segment_id: int = 0
    rotation_deg: float = 0.0
    rotation_confidence: float = 0.0
    rotation_source: str = "none"

    @property
    def has_center(self) -> bool:
        return self.center_x is not None and self.center_y is not None

    @property
    def usable_for_crop(self) -> bool:
        return self.status in {"ok", "clipped", "estimated"} and self.has_center

    @property
    def fully_visible_for_crop(self) -> bool:
        return self.status == "ok" and self.has_center

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameDetection":
        return cls(**data)


def metadata_document(
    input_pattern: str,
    detections: list[FrameDetection],
    crop: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input": input_pattern,
        "crop": crop,
        "frames": [d.to_dict() for d in detections],
    }


def detection_by_name(detections: list[FrameDetection]) -> dict[str, FrameDetection]:
    return {Path(d.filename).name: d for d in detections}
