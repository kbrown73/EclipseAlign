from __future__ import annotations

import argparse
import csv
import json
import shlex
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import astroio
from astroio.exr import write_exr as write_astroio_exr
from tqdm import tqdm

from .detect import DEFAULT_THRESHOLD, DetectionConfig, EllipseFit, detect_file, fit_horizon_ellipse_file
from .diagnostics import write_overlay_preview
from .dust import DustConfig, DustDetectionResult, analyze_dust_inputs
from .files import discover_inputs
from .models import FrameDetection, detection_by_name, metadata_document
from .polish import PolishConfig, polish_aligned_outputs
from .render import compute_centered_square_crop, compute_safe_crop, parse_manual_crop, render_frame
from .rotation import RotationConfig, estimate_rotations
from .track import (
    MIN_RELIABLE_CONFIDENCE,
    MIN_RELIABLE_DISTORTED_CONFIDENCE,
    assign_translations,
    estimate_common_radius,
    refine_detections,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eclipse_align")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="detect eclipse centers and write metadata")
    add_detection_args(detect)
    detect.add_argument("--metadata", required=True, help="metadata JSON output path")
    detect.add_argument("--previews", help="optional diagnostic preview directory")

    render = subparsers.add_parser("render", help="render aligned EXR frames from metadata")
    render.add_argument("--input", required=True, help="input glob or directory")
    render.add_argument("--metadata", required=True, help="metadata JSON from detect")
    render.add_argument("--output", required=True, help="aligned EXR output directory")
    render.add_argument("--jobs", type=int, default=1, help="parallel worker processes; use 0 for all CPUs")
    render.add_argument(
        "--crop",
        action="store_true",
        help="crop to a centered square around the aligned eclipse",
    )
    render.add_argument("--margin", type=int, default=0, help="extra pixels to keep around centered square crop")
    render.add_argument("--manual-crop", help="explicit crop rectangle as WxH+X+Y")
    render.add_argument("--no-rotation", action="store_true", help="ignore rotation metadata during render")
    render.add_argument(
        "--reformat-output",
        action="store_true",
        help="write rendered frames as frame_0001.exr, frame_0002.exr, ...",
    )
    render.add_argument(
        "--alpha-circle",
        action="store_true",
        help="add a filled fitted-sun disk as the output alpha channel",
    )
    add_dust_correction_args(render)
    add_polish_args(render)

    process = subparsers.add_parser("process", help="detect and render in one pass")
    add_detection_args(process)
    process.add_argument("--output", required=True, help="aligned EXR output directory")
    process.add_argument("--diagnostics", required=True, help="diagnostic output directory")
    process.add_argument(
        "--crop",
        action="store_true",
        help="crop to a centered square around the aligned eclipse",
    )
    process.add_argument("--margin", type=int, default=0, help="extra pixels to keep around centered square crop")
    process.add_argument("--manual-crop", help="explicit crop rectangle as WxH+X+Y")
    process.add_argument(
        "--reformat-output",
        action="store_true",
        help="write rendered frames as frame_0001.exr, frame_0002.exr, ...",
    )
    process.add_argument(
        "--alpha-circle",
        action="store_true",
        help="add a filled fitted-sun disk as the output alpha channel",
    )
    add_dust_correction_args(process)
    add_polish_args(process)

    extract_video = subparsers.add_parser("extract-video", help="decode a video into an EXR frame sequence")
    extract_video.add_argument(
        "--input",
        required=True,
        action="append",
        nargs="+",
        help="input video path(s), in chronological order; may be repeated",
    )
    extract_video.add_argument("--output", required=True, help="output EXR frame directory")
    extract_video.add_argument(
        "--output-format",
        default="auto",
        help="AstroIO video decode format: auto, rgb24, rgb48le, gray, or gray16le",
    )
    extract_video.add_argument(
        "--exr-pixel-type",
        choices=("auto", "half", "float"),
        default="auto",
        help="EXR output pixel type; auto writes FLOAT for uint16 frames and HALF otherwise",
    )
    extract_video.add_argument(
        "--transfer",
        choices=("srgb", "none"),
        default="srgb",
        help="transfer conversion before writing EXR; srgb converts display RGB to linear values",
    )
    extract_video.add_argument("--digits", type=int, default=6, help="frame number digits for output filenames")

    return parser


def add_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="input glob or directory")
    parser.add_argument("--work-max-dim", type=int, default=1400, help="max dimension for detection pass")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="normalized bright mask threshold")
    parser.add_argument("--jobs", type=int, default=1, help="parallel worker processes; use 0 for all CPUs")
    parser.add_argument("--detect-rotation", action="store_true", help="estimate one roll correction per reframe segment")
    parser.add_argument("--detect-dust", action="store_true", help="write sensor-fixed dust candidate diagnostics")
    parser.add_argument(
        "--dust-disk-radius",
        type=float,
        default=1.03,
        help="solar radius fraction to inspect for dust diagnostics",
    )
    parser.add_argument(
        "--dust-min-deficit",
        type=float,
        default=0.030,
        help="minimum local dark deficit for per-frame dust candidates",
    )
    parser.add_argument(
        "--dust-min-hit-fraction",
        type=float,
        default=0.10,
        help="minimum repeated-hit fraction for aggregate dust candidates",
    )
    parser.add_argument(
        "--dust-min-support-frames",
        type=int,
        default=6,
        help="minimum supported frames for aggregate dust candidates",
    )
    parser.add_argument("--rotation-jump-threshold", type=float, default=120.0, help="raw center jump threshold for reframe detection")
    parser.add_argument(
        "--rotation-jobs",
        type=int,
        default=None,
        help="parallel worker processes for rotation estimation; defaults to --jobs",
    )
    parser.add_argument(
        "--preview-max-dim",
        type=int,
        default=1600,
        help="max dimension for diagnostic previews; use 0 for full resolution",
    )
    parser.add_argument(
        "--prefer-plausible-raw",
        action="store_true",
        help="use plausible low-confidence raw fits instead of interpolation across all frames",
    )
    parser.add_argument(
        "--plausible-raw-range",
        action="append",
        default=[],
        metavar="RANGE[,RANGE...]",
        help=(
            "limit plausible raw fit overrides to comma-separated 1-based inclusive frame ranges; "
            "may be repeated, e.g. 1225-1300,2000-2100,3720+"
        ),
    )
    parser.add_argument(
        "--horizon-ellipse-range",
        action="append",
        default=[],
        metavar="RANGE[,RANGE...]",
        help=(
            "fit and prefer guarded ellipse centers for reviewed horizon/distortion frame ranges; "
            "uses 1-based inclusive frame numbers and may be repeated, e.g. 1225-1300,3720+"
        ),
    )


def add_polish_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--polish-alignment",
        action="store_true",
        help="apply a bounded post-render residual translation polish pass",
    )
    parser.add_argument(
        "--polish-max-shift",
        type=float,
        default=2.0,
        help="maximum residual polish correction in pixels",
    )


def add_dust_correction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--correct-dust",
        action="store_true",
        help="apply the sensor-fixed dust mask before alignment/rendering",
    )
    parser.add_argument(
        "--dust-mask",
        help="dust mask PNG to use with --correct-dust; defaults to dust/dust_mask.png next to metadata/diagnostics",
    )
    parser.add_argument(
        "--dust-correction-radius",
        type=float,
        default=5.0,
        help="inpaint radius in source pixels for --correct-dust",
    )
    parser.add_argument(
        "--dust-mask-dilation",
        type=int,
        default=2,
        help="source-pixel dilation applied to the dust mask before correction",
    )


def dust_config(args: argparse.Namespace) -> DustConfig:
    if args.dust_disk_radius <= 0:
        raise SystemExit("--dust-disk-radius must be greater than zero")
    if args.dust_min_deficit <= 0:
        raise SystemExit("--dust-min-deficit must be greater than zero")
    if not 0 < args.dust_min_hit_fraction <= 1:
        raise SystemExit("--dust-min-hit-fraction must be greater than zero and no more than one")
    if args.dust_min_support_frames < 1:
        raise SystemExit("--dust-min-support-frames must be at least one")
    return DustConfig(
        work_max_dim=args.work_max_dim,
        disk_radius_fraction=args.dust_disk_radius,
        min_deficit=args.dust_min_deficit,
        min_hit_fraction=args.dust_min_hit_fraction,
        min_support_frames=args.dust_min_support_frames,
    )


def detection_config(args: argparse.Namespace) -> DetectionConfig:
    return DetectionConfig(work_max_dim=args.work_max_dim, threshold=args.threshold)


def resolve_jobs(value: int) -> int:
    if value < 0:
        raise SystemExit("--jobs must be zero or greater")
    if value == 0:
        return os.cpu_count() or 1
    return value


def resolve_rotation_jobs(args: argparse.Namespace) -> int:
    if args.rotation_jobs is not None:
        return resolve_jobs(args.rotation_jobs)
    # Boundary registration reads two full EXRs per task. Keep the default
    # conservative even when detection uses --jobs 0.
    return max(1, resolve_jobs(args.jobs) - 2)


def resolve_preview_jobs(args: argparse.Namespace) -> int:
    # Preview generation is I/O and memory heavy because every worker reads a
    # full EXR and writes a PNG. With --jobs 0, leave a couple of cores free so
    # the machine remains responsive while still scaling with available CPUs.
    return max(1, resolve_jobs(args.jobs) - 2)


def process_pool_context():
    return mp.get_context("spawn")


def parallel_map_ordered(fn, items, *, jobs: int, desc: str):
    if jobs == 1:
        return [fn(item) for item in tqdm(items, desc=desc)]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=process_pool_context()) as executor:
        return list(tqdm(executor.map(fn, items), total=len(items), desc=desc))


def parallel_run_unordered(fn, items, *, jobs: int, desc: str) -> None:
    if jobs == 1:
        for item in tqdm(items, desc=desc):
            fn(item)
        return
    with ProcessPoolExecutor(max_workers=jobs, mp_context=process_pool_context()) as executor:
        futures = [executor.submit(fn, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            future.result()


def detect_worker(payload: tuple[str, DetectionConfig, float | None]) -> FrameDetection:
    path, config, expected_radius = payload
    return detect_file(path, config, expected_radius=expected_radius)


def horizon_ellipse_worker(payload: tuple[str, DetectionConfig, float | None]) -> EllipseFit | None:
    path, config, expected_radius = payload
    return fit_horizon_ellipse_file(path, config, expected_radius=expected_radius)


def preview_worker(payload: tuple[str, FrameDetection, DetectionConfig, str, int]) -> None:
    input_path, detection, config, preview_path, preview_max_dim = payload
    write_overlay_preview(input_path, preview_path, detection, config, max_dim=preview_max_dim)


RenderPayload = tuple[str, str, FrameDetection, object, bool, bool, str | None, float, int]


def render_worker(payload: RenderPayload) -> bool:
    (
        input_path,
        output_path,
        detection,
        crop,
        apply_rotation,
        add_alpha_circle,
        dust_mask_path,
        dust_correction_radius,
        dust_mask_dilation,
    ) = payload
    if detection.translation_x is None or detection.translation_y is None:
        return False
    render_frame(
        input_path,
        output_path,
        detection,
        crop=crop,
        apply_rotation=apply_rotation,
        add_alpha_circle=add_alpha_circle,
        dust_mask_path=dust_mask_path,
        dust_correction_radius=dust_correction_radius,
        dust_mask_dilation=dust_mask_dilation,
    )
    return True


def needs_radius_constrained_retry(detection: FrameDetection) -> bool:
    retry_flags = {"radius_outlier", "insufficient_limb"}
    if detection.status in {"failed", "low_confidence"} or any(flag in retry_flags for flag in detection.flags):
        return True
    return (
        "distorted_limb_suspected" in detection.flags
        and detection.confidence < MIN_RELIABLE_DISTORTED_CONFIDENCE
    )


def retry_improves_detection(
    original: FrameDetection,
    candidate: FrameDetection,
    common_radius: float,
) -> bool:
    if not candidate.has_center or candidate.radius is None:
        return False
    if candidate.status not in {"ok", "clipped"}:
        return False
    if "distorted_limb_suspected" in candidate.flags or "radius_outlier" in candidate.flags:
        return False
    radius_error = abs(candidate.radius - common_radius) / max(common_radius, 1.0)
    if radius_error > 0.20:
        return False

    original_residual = original.circle_residual_median_px
    candidate_residual = candidate.circle_residual_median_px
    if original_residual is not None and candidate_residual is not None and candidate_residual > original_residual:
        return False

    if candidate.confidence >= MIN_RELIABLE_CONFIDENCE:
        return True
    return candidate.confidence > original.confidence


def improve_suspicious_detections(
    inputs: list[Path],
    detections: list[FrameDetection],
    config: DetectionConfig,
    *,
    jobs: int,
) -> tuple[float | None, int]:
    common_radius = estimate_common_radius(detections)
    if common_radius is None:
        return None, 0

    retry_indexes = [
        idx
        for idx, detection in enumerate(detections)
        if needs_radius_constrained_retry(detection)
    ]
    if not retry_indexes:
        return common_radius, 0

    retries = parallel_map_ordered(
        detect_worker,
        [(str(inputs[idx]), config, common_radius) for idx in retry_indexes],
        jobs=jobs,
        desc="redetect",
    )

    improved = 0
    for idx, retry in zip(retry_indexes, retries):
        if not retry_improves_detection(detections[idx], retry, common_radius):
            continue
        retry.flags.append("radius_constrained_redetect")
        detections[idx] = retry
        improved += 1
    return estimate_common_radius(detections), improved


PLAUSIBLE_RAW_RADIUS_TOLERANCE = 0.25
PLAUSIBLE_RAW_MIN_CONFIDENCE = 0.08
PLAUSIBLE_RAW_MIN_SUPPORT = 0.35
PLAUSIBLE_RAW_MAX_MEDIAN_RESIDUAL_PX = 50.0


def parse_frame_ranges(values: list[str], frame_count: int, option_name: str) -> set[int]:
    indexes: set[int] = set()
    for value in values:
        for part in value.split(","):
            token = part.strip()
            if not token:
                continue
            if token.endswith("+"):
                start_text = token[:-1]
                end_text = str(frame_count)
            elif "-" in token:
                start_text, end_text = token.split("-", 1)
            else:
                start_text = token
                end_text = token
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise SystemExit(f"Invalid {option_name} value: {token}") from exc
            if start < 1 or end < start or end > frame_count:
                raise SystemExit(
                    f"{option_name} must be within 1-{frame_count} and ordered: {token}"
                )
            indexes.update(range(start - 1, end))
    return indexes


def parse_plausible_raw_ranges(values: list[str], frame_count: int) -> set[int]:
    return parse_frame_ranges(values, frame_count, "--plausible-raw-range")


def plausible_raw_indexes(args: argparse.Namespace, frame_count: int) -> set[int]:
    ranges = parse_plausible_raw_ranges(args.plausible_raw_range, frame_count)
    if ranges:
        return ranges
    if args.prefer_plausible_raw:
        return set(range(frame_count))
    return set()


def horizon_ellipse_indexes(args: argparse.Namespace, frame_count: int) -> set[int]:
    return parse_frame_ranges(args.horizon_ellipse_range, frame_count, "--horizon-ellipse-range")


def is_plausible_raw_fit(detection: FrameDetection, common_radius: float | None) -> bool:
    if common_radius is None:
        return False
    if detection.raw_center_x is None or detection.raw_center_y is None or detection.raw_radius is None:
        return False
    if detection.status != "estimated" and "interpolated" not in detection.flags:
        return False
    if detection.confidence < PLAUSIBLE_RAW_MIN_CONFIDENCE:
        return False
    if (
        detection.limb_support_fraction is None
        or detection.limb_support_fraction < PLAUSIBLE_RAW_MIN_SUPPORT
    ):
        return False
    if (
        detection.circle_residual_median_px is None
        or detection.circle_residual_median_px > PLAUSIBLE_RAW_MAX_MEDIAN_RESIDUAL_PX
    ):
        return False
    radius_error = abs(detection.raw_radius - common_radius) / max(common_radius, 1.0)
    return radius_error <= PLAUSIBLE_RAW_RADIUS_TOLERANCE


def apply_plausible_raw_overrides(
    detections: list[FrameDetection],
    indexes: set[int],
    common_radius: float | None,
) -> int:
    applied = 0
    for idx in sorted(indexes):
        detection = detections[idx]
        if not is_plausible_raw_fit(detection, common_radius):
            continue
        detection.center_x = detection.raw_center_x
        detection.center_y = detection.raw_center_y
        detection.radius = detection.raw_radius
        detection.status = "estimated"
        detection.flags = [flag for flag in detection.flags if flag != "interpolated"]
        if "raw_fit_override" not in detection.flags:
            detection.flags.append("raw_fit_override")
        applied += 1
    if applied:
        assign_translations(detections)
    return applied


HORIZON_ELLIPSE_RADIUS_TOLERANCE = 0.35
HORIZON_ELLIPSE_MAX_CENTER_DRIFT_FRACTION = 0.35
HORIZON_ELLIPSE_MAX_CENTER_DRIFT_PX = 90.0


def ellipse_fit_improves_detection(
    detection: FrameDetection,
    ellipse: EllipseFit,
    common_radius: float | None,
    config: DetectionConfig,
) -> bool:
    if ellipse.residual_median_px > config.max_ellipse_median_residual_px:
        return False
    if ellipse.support_fraction < config.distorted_limb_support_fraction:
        return False

    reference_radius = common_radius if common_radius is not None else detection.radius
    if reference_radius is not None:
        radius_error = abs(ellipse.major_radius - reference_radius) / max(reference_radius, 1.0)
        if radius_error > HORIZON_ELLIPSE_RADIUS_TOLERANCE:
            return False

    if detection.has_center:
        center_drift = float(np.hypot(ellipse.center_x - detection.center_x, ellipse.center_y - detection.center_y))
        if reference_radius is not None:
            max_drift = min(
                HORIZON_ELLIPSE_MAX_CENTER_DRIFT_PX,
                max(24.0, reference_radius * HORIZON_ELLIPSE_MAX_CENTER_DRIFT_FRACTION),
            )
        else:
            max_drift = HORIZON_ELLIPSE_MAX_CENTER_DRIFT_PX
        if center_drift > max_drift:
            return False

    if detection.circle_residual_median_px is not None:
        required = detection.circle_residual_median_px * (1.0 - config.min_ellipse_improvement)
        if ellipse.residual_median_px >= required:
            return False

    return True


def apply_horizon_ellipse_overrides(
    inputs: list[Path],
    detections: list[FrameDetection],
    indexes: set[int],
    config: DetectionConfig,
    common_radius: float | None,
    *,
    jobs: int,
) -> int:
    if not indexes:
        return 0

    sorted_indexes = sorted(indexes)
    fits = parallel_map_ordered(
        horizon_ellipse_worker,
        [(str(inputs[idx]), config, common_radius) for idx in sorted_indexes],
        jobs=jobs,
        desc="ellipse",
    )

    applied = 0
    for idx, ellipse in zip(sorted_indexes, fits):
        if ellipse is None:
            continue
        detection = detections[idx]
        if not ellipse_fit_improves_detection(detection, ellipse, common_radius, config):
            continue

        detection.ellipse_center_x = ellipse.center_x
        detection.ellipse_center_y = ellipse.center_y
        detection.ellipse_major_radius = ellipse.major_radius
        detection.ellipse_minor_radius = ellipse.minor_radius
        detection.ellipse_angle_deg = ellipse.angle_deg
        detection.ellipse_residual_median_px = ellipse.residual_median_px
        detection.ellipse_residual_p90_px = ellipse.residual_p90_px
        detection.ellipse_support_fraction = ellipse.support_fraction
        detection.center_x = ellipse.center_x
        detection.center_y = ellipse.center_y
        detection.radius = ellipse.major_radius
        detection.confidence = min(0.75, max(0.35, ellipse.confidence))
        detection.status = "estimated"
        detection.flags = [flag for flag in detection.flags if flag != "interpolated"]
        if "horizon_ellipse_fit" not in detection.flags:
            detection.flags.append("horizon_ellipse_fit")
        applied += 1

    if applied:
        assign_translations(detections)
    return applied


def reformatted_output_filename(frame_number: int) -> str:
    if frame_number < 1:
        raise ValueError("frame_number must start at 1")
    return f"frame_{frame_number:04d}.exr"


def video_frame_filename(frame_number: int, *, digits: int = 6) -> str:
    if frame_number < 1:
        raise ValueError("frame_number must start at 1")
    if digits < 1:
        raise ValueError("digits must be at least one")
    return f"frame_{frame_number:0{digits}d}.exr"


def exr_half_for_video_frame(frame: np.ndarray, exr_pixel_type: str) -> bool:
    if exr_pixel_type == "half":
        return True
    if exr_pixel_type == "float":
        return False
    if exr_pixel_type != "auto":
        raise ValueError(f"Unknown EXR pixel type: {exr_pixel_type}")
    return frame.dtype != np.uint16


def video_frame_to_exr_float(frame: np.ndarray, *, transfer: str = "srgb") -> np.ndarray:
    image = np.asarray(frame)
    if image.dtype == np.uint8:
        normalized = image.astype(np.float32) / np.float32(255.0)
    elif image.dtype == np.uint16:
        normalized = image.astype(np.float32) / np.float32(65535.0)
    else:
        normalized = image.astype(np.float32, copy=False)

    if transfer == "none":
        return normalized
    if transfer == "srgb":
        return srgb_to_linear(normalized)
    raise ValueError(f"Unknown transfer conversion: {transfer}")


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def normalize_video_inputs(input_paths: str | Path | Iterable[str | Path | Iterable[str | Path]]) -> list[Path]:
    paths: list[Path] = []

    def add_input(raw_input: str | Path) -> None:
        if isinstance(raw_input, Path):
            paths.append(raw_input)
            return

        input_text = raw_input.strip()
        input_path = Path(input_text)
        if not input_text:
            return
        if input_path.exists() or not any(char.isspace() for char in input_text):
            paths.append(input_path)
            return
        paths.extend(Path(part) for part in shlex.split(input_text))

    if isinstance(input_paths, (str, Path)):
        add_input(input_paths)
    else:
        for raw_input in input_paths:
            if isinstance(raw_input, (str, Path)):
                add_input(raw_input)
            else:
                for part in raw_input:
                    add_input(part)

    if not paths:
        raise ValueError("at least one input video path is required")
    return paths


def extract_video_frames(
    input_path: str | Path | Iterable[str | Path | Iterable[str | Path]],
    output_dir: str | Path,
    *,
    output_format: str = "auto",
    exr_pixel_type: str = "auto",
    transfer: str = "srgb",
    digits: int = 6,
) -> int:
    input_paths = normalize_video_inputs(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_number = 1
    total_count = 0
    for path in input_paths:
        desc = "extract-video" if len(input_paths) == 1 else f"extract-video {path.name}"
        with astroio.open_reader(path, output_format=output_format) as reader:
            written = 0
            for frame in tqdm(reader, total=reader.frame_count, desc=desc):
                output_path = output_dir / video_frame_filename(frame_number, digits=digits)
                write_astroio_exr(
                    output_path,
                    video_frame_to_exr_float(frame, transfer=transfer),
                    half=exr_half_for_video_frame(frame, exr_pixel_type),
                )
                frame_number += 1
                written += 1
            total_count += written
    return total_count


def build_render_payloads(
    inputs: list[Path],
    detections_by_name: dict[str, FrameDetection],
    output_dir: Path,
    crop: object,
    apply_rotation: bool,
    *,
    reformat_output: bool,
    add_alpha_circle: bool,
    dust_mask_path: Path | None = None,
    dust_correction_radius: float = 5.0,
    dust_mask_dilation: int = 2,
) -> tuple[list[RenderPayload], int, dict[str, str]]:
    skipped = 0
    payloads = []
    output_name_by_source: dict[str, str] = {}
    frame_number = 1
    for path in inputs:
        detection = detections_by_name.get(path.name)
        if detection is None or detection.translation_x is None or detection.translation_y is None:
            skipped += 1
            continue
        output_name = reformatted_output_filename(frame_number) if reformat_output else path.name
        frame_number += 1
        output_name_by_source[path.name] = output_name
        payloads.append(
            (
                str(path),
                str(output_dir / output_name),
                detection,
                crop,
                apply_rotation,
                add_alpha_circle,
                str(dust_mask_path) if dust_mask_path is not None else None,
                dust_correction_radius,
                dust_mask_dilation,
            )
        )
    return payloads, skipped, output_name_by_source


def detections_with_output_filenames(
    detections: list[FrameDetection],
    output_name_by_source: dict[str, str],
) -> list[FrameDetection]:
    return [
        replace(detection, filename=output_name_by_source.get(Path(detection.filename).name, detection.filename))
        for detection in detections
    ]


def maybe_polish_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    detections: list[FrameDetection],
    summary_path: Path,
) -> int:
    if not args.polish_alignment:
        return 0
    config = PolishConfig(max_shift_px=args.polish_max_shift)
    polish_results = polish_aligned_outputs(output_dir, detections, config=config)
    write_polish_summary(summary_path, polish_results)
    return sum(1 for result in polish_results if result.source == "phase_correlation")


def resolve_dust_mask_path(args: argparse.Namespace, default_path: Path | None) -> Path | None:
    if not args.correct_dust:
        if args.dust_mask:
            raise SystemExit("--dust-mask requires --correct-dust")
        return None
    if args.dust_correction_radius <= 0:
        raise SystemExit("--dust-correction-radius must be greater than zero")
    if args.dust_mask_dilation < 0:
        raise SystemExit("--dust-mask-dilation must be zero or greater")

    path = Path(args.dust_mask) if args.dust_mask else default_path
    if path is None:
        raise SystemExit("--correct-dust requires --dust-mask or a dust mask from --detect-dust")
    if not path.exists():
        raise SystemExit(f"Dust mask not found: {path}")
    return path


def load_metadata(path: str | Path) -> tuple[dict, list[FrameDetection]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data, [FrameDetection.from_dict(item) for item in data["frames"]]


def write_metadata(path: str | Path, document: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def frame_index_by_segment(detections: list[FrameDetection]) -> dict[int, tuple[int, int, str, str]]:
    ranges: dict[int, list] = {}
    for idx, detection in enumerate(detections):
        if detection.segment_id not in ranges:
            ranges[detection.segment_id] = [idx, idx, detection.filename, detection.filename]
        else:
            ranges[detection.segment_id][1] = idx
            ranges[detection.segment_id][3] = detection.filename
    return {segment_id: tuple(values) for segment_id, values in ranges.items()}


def write_rotation_summary(
    path: str | Path,
    detections: list[FrameDetection],
    rotation_boundaries: list,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    segments = frame_index_by_segment(detections)
    fieldnames = [
        "boundary_index",
        "from_segment",
        "to_segment",
        "before_frame_index",
        "after_frame_index",
        "before_filename",
        "after_filename",
        "delta_deg",
        "cumulative_rotation_deg",
        "confidence",
        "source",
        "score",
        "from_drift_angle_deg",
        "to_drift_angle_deg",
        "raw_drift_delta_deg",
        "expected_drift_delta_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for boundary in rotation_boundaries:
            before_segment = segments.get(boundary.from_segment)
            after_segment = segments.get(boundary.to_segment)
            after_detection = detections[after_segment[0]] if after_segment is not None else None
            writer.writerow(
                {
                    "boundary_index": boundary.boundary_index,
                    "from_segment": boundary.from_segment,
                    "to_segment": boundary.to_segment,
                    "before_frame_index": before_segment[1] if before_segment is not None else "",
                    "after_frame_index": after_segment[0] if after_segment is not None else "",
                    "before_filename": before_segment[3] if before_segment is not None else "",
                    "after_filename": after_segment[2] if after_segment is not None else "",
                    "delta_deg": boundary.delta_deg,
                    "cumulative_rotation_deg": (
                        after_detection.rotation_deg if after_detection is not None else ""
                    ),
                    "confidence": boundary.confidence,
                    "source": boundary.source,
                    "score": boundary.score,
                    "from_drift_angle_deg": none_as_empty(boundary.from_drift_angle_deg),
                    "to_drift_angle_deg": none_as_empty(boundary.to_drift_angle_deg),
                    "raw_drift_delta_deg": none_as_empty(boundary.raw_drift_delta_deg),
                    "expected_drift_delta_deg": none_as_empty(boundary.expected_drift_delta_deg),
                }
            )


def none_as_empty(value):
    return "" if value is None else value


def write_polish_summary(path: str | Path, polish_results: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "filename",
        "segment_id",
        "residual_dx",
        "residual_dy",
        "confidence",
        "score",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in polish_results:
            writer.writerow(result.__dict__)


def dust_metadata(result: DustDetectionResult) -> dict:
    return {
        "frames_used": result.frames_used,
        "map_width": result.map_width,
        "map_height": result.map_height,
        "scale": result.scale,
        "components": [component.__dict__ for component in result.components],
    }


def summarize(detections: Iterable[FrameDetection]) -> str:
    counts = Counter(d.status for d in detections)
    parts = [f"{name}={counts[name]}" for name in sorted(counts)]
    return ", ".join(parts)


def detect_command(args: argparse.Namespace) -> int:
    config = detection_config(args)
    jobs = resolve_jobs(args.jobs)
    inputs = discover_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No EXR inputs matched: {args.input}")

    detections = parallel_map_ordered(
        detect_worker,
        [(str(path), config, None) for path in inputs],
        jobs=jobs,
        desc="detect",
    )
    common_radius, redetected = improve_suspicious_detections(inputs, detections, config, jobs=jobs)
    common_radius = refine_detections(detections)
    raw_overrides = apply_plausible_raw_overrides(
        detections,
        plausible_raw_indexes(args, len(detections)),
        common_radius,
    )
    ellipse_overrides = apply_horizon_ellipse_overrides(
        inputs,
        detections,
        horizon_ellipse_indexes(args, len(detections)),
        config,
        common_radius,
        jobs=jobs,
    )
    rotation_boundaries = []
    if args.detect_rotation:
        rotation_boundaries = estimate_rotations(
            inputs,
            detections,
            config=RotationConfig(jump_threshold_px=args.rotation_jump_threshold),
            map_fn=lambda fn, items, desc: parallel_map_ordered(
                fn,
                items,
                jobs=resolve_rotation_jobs(args),
                desc=desc,
            ),
        )
    dust_result = None
    if args.detect_dust:
        dust_result = analyze_dust_inputs(
            inputs,
            detections,
            Path(args.metadata).with_name("dust"),
            config=dust_config(args),
            preview_max_dim=args.preview_max_dim,
        )

    document = metadata_document(
        args.input,
        detections,
        crop=None,
    )
    document["common_radius"] = common_radius
    document["rotation_boundaries"] = [boundary.__dict__ for boundary in rotation_boundaries]
    if dust_result is not None:
        document["dust"] = dust_metadata(dust_result)
    write_metadata(args.metadata, document)
    if rotation_boundaries:
        write_rotation_summary(Path(args.metadata).with_name("rotation_summary.csv"), detections, rotation_boundaries)

    if args.previews:
        write_previews(
            inputs,
            detections,
            Path(args.previews),
            config,
            args.preview_max_dim,
            resolve_preview_jobs(args),
        )

    print(f"Detected {len(detections)} frames: {summarize(detections)}")
    if redetected:
        print(f"Radius-constrained redetect improved {redetected} frames")
    if raw_overrides:
        print(f"Plausible raw fit overrides applied to {raw_overrides} frames")
    if ellipse_overrides:
        print(f"Horizon ellipse overrides applied to {ellipse_overrides} frames")
    if common_radius is not None:
        print(f"Common radius estimate: {common_radius:.2f}px")
    if dust_result is not None:
        print(f"Dust candidates: {len(dust_result.components)}")
        print(f"Dust diagnostics: {Path(args.metadata).with_name('dust')}")
    return 0


def resolve_crop(args: argparse.Namespace, detections: list[FrameDetection]):
    if args.manual_crop and args.crop:
        raise SystemExit("Use either --crop or --manual-crop, not both")
    if args.manual_crop:
        return parse_manual_crop(args.manual_crop)
    if not args.crop:
        if args.margin:
            raise SystemExit("--margin requires --crop")
        return None
    try:
        crop = compute_centered_square_crop(detections, margin=args.margin)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if crop is None:
        raise SystemExit("Could not compute centered square crop")
    return crop


def render_command(args: argparse.Namespace) -> int:
    jobs = resolve_jobs(args.jobs)
    inputs = discover_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No EXR inputs matched: {args.input}")

    _, detections = load_metadata(args.metadata)
    by_name = detection_by_name(detections)
    crop = resolve_crop(args, detections)
    apply_rotation = not args.no_rotation
    output_dir = Path(args.output)
    dust_mask_path = resolve_dust_mask_path(
        args,
        Path(args.metadata).with_name("dust") / "dust_mask.png",
    )
    payloads, skipped, output_name_by_source = build_render_payloads(
        inputs,
        by_name,
        output_dir,
        crop,
        apply_rotation,
        reformat_output=args.reformat_output,
        add_alpha_circle=args.alpha_circle,
        dust_mask_path=dust_mask_path,
        dust_correction_radius=args.dust_correction_radius,
        dust_mask_dilation=args.dust_mask_dilation,
    )
    rendered = sum(
        1
        for result in parallel_map_ordered(render_worker, payloads, jobs=jobs, desc="render")
        if result
    )
    print(f"Rendered {rendered} frames to {output_dir}")
    polished = maybe_polish_outputs(
        args,
        output_dir,
        detections_with_output_filenames(detections, output_name_by_source),
        Path(args.metadata).with_name("polish_summary.csv"),
    )
    if args.polish_alignment:
        print(f"Polished {polished} frames")
        print(f"Polish summary: {Path(args.metadata).with_name('polish_summary.csv')}")
    if skipped:
        print(f"Skipped {skipped} frames without usable metadata")
    if crop is not None:
        print(f"Crop: {crop.width}x{crop.height}+{crop.left}+{crop.top}")
    if dust_mask_path is not None:
        print(f"Dust correction mask: {dust_mask_path}")
    return 0


def process_command(args: argparse.Namespace) -> int:
    config = detection_config(args)
    jobs = resolve_jobs(args.jobs)
    inputs = discover_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No EXR inputs matched: {args.input}")

    diagnostics_dir = Path(args.diagnostics)
    metadata_path = diagnostics_dir / "detections.json"
    rotation_summary_path = diagnostics_dir / "rotation_summary.csv"
    polish_summary_path = diagnostics_dir / "polish_summary.csv"
    dust_dir = diagnostics_dir / "dust"
    previews_dir = diagnostics_dir / "previews"

    detections = parallel_map_ordered(
        detect_worker,
        [(str(path), config, None) for path in inputs],
        jobs=jobs,
        desc="detect",
    )
    common_radius, redetected = improve_suspicious_detections(inputs, detections, config, jobs=jobs)
    common_radius = refine_detections(detections)
    raw_overrides = apply_plausible_raw_overrides(
        detections,
        plausible_raw_indexes(args, len(detections)),
        common_radius,
    )
    ellipse_overrides = apply_horizon_ellipse_overrides(
        inputs,
        detections,
        horizon_ellipse_indexes(args, len(detections)),
        config,
        common_radius,
        jobs=jobs,
    )
    rotation_boundaries = []
    if args.detect_rotation:
        rotation_boundaries = estimate_rotations(
            inputs,
            detections,
            config=RotationConfig(jump_threshold_px=args.rotation_jump_threshold),
            map_fn=lambda fn, items, desc: parallel_map_ordered(
                fn,
                items,
                jobs=resolve_rotation_jobs(args),
                desc=desc,
            ),
        )
    dust_result = None
    if args.detect_dust:
        dust_result = analyze_dust_inputs(
            inputs,
            detections,
            dust_dir,
            config=dust_config(args),
            preview_max_dim=args.preview_max_dim,
        )
    dust_mask_path = resolve_dust_mask_path(
        args,
        dust_dir / "dust_mask.png" if args.detect_dust else None,
    )
    crop = resolve_crop(args, detections)
    crop_data = crop.to_dict() if crop is not None else None
    document = metadata_document(args.input, detections, crop=crop_data)
    document["common_radius"] = common_radius
    document["rotation_boundaries"] = [boundary.__dict__ for boundary in rotation_boundaries]
    if dust_result is not None:
        document["dust"] = dust_metadata(dust_result)
    write_metadata(metadata_path, document)
    if rotation_boundaries:
        write_rotation_summary(rotation_summary_path, detections, rotation_boundaries)

    write_previews(inputs, detections, previews_dir, config, args.preview_max_dim, resolve_preview_jobs(args))

    output_dir = Path(args.output)
    by_name = detection_by_name(detections)
    payloads, _, output_name_by_source = build_render_payloads(
        inputs,
        by_name,
        output_dir,
        crop,
        True,
        reformat_output=args.reformat_output,
        add_alpha_circle=args.alpha_circle,
        dust_mask_path=dust_mask_path,
        dust_correction_radius=args.dust_correction_radius,
        dust_mask_dilation=args.dust_mask_dilation,
    )
    rendered = sum(
        1
        for result in parallel_map_ordered(render_worker, payloads, jobs=jobs, desc="render")
        if result
    )
    polished = maybe_polish_outputs(
        args,
        output_dir,
        detections_with_output_filenames(detections, output_name_by_source),
        polish_summary_path,
    )

    print(f"Detected {len(detections)} frames: {summarize(detections)}")
    if redetected:
        print(f"Radius-constrained redetect improved {redetected} frames")
    if raw_overrides:
        print(f"Plausible raw fit overrides applied to {raw_overrides} frames")
    if ellipse_overrides:
        print(f"Horizon ellipse overrides applied to {ellipse_overrides} frames")
    print(f"Rendered {rendered} frames to {output_dir}")
    if args.polish_alignment:
        print(f"Polished {polished} frames")
    print(f"Metadata: {metadata_path}")
    if rotation_boundaries:
        print(f"Rotation summary: {rotation_summary_path}")
    if args.polish_alignment:
        print(f"Polish summary: {polish_summary_path}")
    if dust_result is not None:
        print(f"Dust candidates: {len(dust_result.components)}")
        print(f"Dust diagnostics: {dust_dir}")
    if dust_mask_path is not None:
        print(f"Dust correction mask: {dust_mask_path}")
    print(f"Previews: {previews_dir}")
    if common_radius is not None:
        print(f"Common radius estimate: {common_radius:.2f}px")
    if crop is not None:
        print(f"Crop: {crop.width}x{crop.height}+{crop.left}+{crop.top}")
    return 0


def extract_video_command(args: argparse.Namespace) -> int:
    count = extract_video_frames(
        args.input,
        args.output,
        output_format=args.output_format,
        exr_pixel_type=args.exr_pixel_type,
        transfer=args.transfer,
        digits=args.digits,
    )
    print(f"Extracted {count} frames to {Path(args.output)}")
    return 0


def write_previews(
    inputs: list[Path],
    detections: list[FrameDetection],
    previews_dir: Path,
    config: DetectionConfig,
    preview_max_dim: int,
    jobs: int,
) -> None:
    previews_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        (str(path), detection, config, str(previews_dir / f"{path.stem}.png"), preview_max_dim)
        for path, detection in zip(inputs, detections)
    ]
    parallel_run_unordered(preview_worker, payloads, jobs=jobs, desc="previews")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "detect":
        return detect_command(args)
    if args.command == "render":
        return render_command(args)
    if args.command == "process":
        return process_command(args)
    if args.command == "extract-video":
        return extract_video_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2
