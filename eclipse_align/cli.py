from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .detect import DetectionConfig, detect_file
from .diagnostics import write_overlay_preview
from .dust import DustConfig, DustDetectionResult, analyze_dust_inputs
from .files import discover_inputs
from .models import FrameDetection, detection_by_name, metadata_document
from .polish import PolishConfig, polish_aligned_outputs
from .render import compute_centered_square_crop, compute_safe_crop, parse_manual_crop, render_frame
from .rotation import RotationConfig, estimate_rotations
from .track import refine_detections


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
    add_polish_args(process)

    return parser


def add_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="input glob or directory")
    parser.add_argument("--work-max-dim", type=int, default=1400, help="max dimension for detection pass")
    parser.add_argument("--threshold", type=float, default=0.18, help="normalized bright mask threshold")
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


def detect_worker(payload: tuple[str, DetectionConfig]) -> FrameDetection:
    path, config = payload
    return detect_file(path, config)


def preview_worker(payload: tuple[str, FrameDetection, DetectionConfig, str, int]) -> None:
    input_path, detection, config, preview_path, preview_max_dim = payload
    write_overlay_preview(input_path, preview_path, detection, config, max_dim=preview_max_dim)


def render_worker(payload: tuple[str, str, FrameDetection, object, bool, bool]) -> bool:
    input_path, output_path, detection, crop, apply_rotation, add_alpha_circle = payload
    if detection.translation_x is None or detection.translation_y is None:
        return False
    render_frame(
        input_path,
        output_path,
        detection,
        crop=crop,
        apply_rotation=apply_rotation,
        add_alpha_circle=add_alpha_circle,
    )
    return True


def reformatted_output_filename(frame_number: int) -> str:
    if frame_number < 1:
        raise ValueError("frame_number must start at 1")
    return f"frame_{frame_number:04d}.exr"


def build_render_payloads(
    inputs: list[Path],
    detections_by_name: dict[str, FrameDetection],
    output_dir: Path,
    crop: object,
    apply_rotation: bool,
    *,
    reformat_output: bool,
    add_alpha_circle: bool,
) -> tuple[list[tuple[str, str, FrameDetection, object, bool, bool]], int, dict[str, str]]:
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
        [(str(path), config) for path in inputs],
        jobs=jobs,
        desc="detect",
    )
    common_radius = refine_detections(detections)
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
    payloads, skipped, output_name_by_source = build_render_payloads(
        inputs,
        by_name,
        output_dir,
        crop,
        apply_rotation,
        reformat_output=args.reformat_output,
        add_alpha_circle=args.alpha_circle,
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
        [(str(path), config) for path in inputs],
        jobs=jobs,
        desc="detect",
    )
    common_radius = refine_detections(detections)
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
    print(f"Previews: {previews_dir}")
    if common_radius is not None:
        print(f"Common radius estimate: {common_radius:.2f}px")
    if crop is not None:
        print(f"Crop: {crop.width}x{crop.height}+{crop.left}+{crop.top}")
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
    parser.error(f"Unknown command: {args.command}")
    return 2
