from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .detect import DetectionConfig, detect_file
from .diagnostics import write_overlay_preview
from .files import discover_inputs
from .models import FrameDetection, detection_by_name, metadata_document
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

    return parser


def add_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="input glob or directory")
    parser.add_argument("--work-max-dim", type=int, default=1400, help="max dimension for detection pass")
    parser.add_argument("--threshold", type=float, default=0.18, help="normalized bright mask threshold")
    parser.add_argument("--jobs", type=int, default=1, help="parallel worker processes; use 0 for all CPUs")
    parser.add_argument("--detect-rotation", action="store_true", help="estimate one roll correction per reframe segment")
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
    return min(resolve_jobs(args.jobs), 4)


def parallel_map_ordered(fn, items, *, jobs: int, desc: str):
    if jobs == 1:
        return [fn(item) for item in tqdm(items, desc=desc)]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(tqdm(executor.map(fn, items), total=len(items), desc=desc))


def detect_worker(payload: tuple[str, DetectionConfig]) -> FrameDetection:
    path, config = payload
    return detect_file(path, config)


def preview_worker(payload: tuple[str, FrameDetection, DetectionConfig, str, int]) -> None:
    input_path, detection, config, preview_path, preview_max_dim = payload
    write_overlay_preview(input_path, preview_path, detection, config, max_dim=preview_max_dim)


def render_worker(payload: tuple[str, str, FrameDetection, object, bool]) -> bool:
    input_path, output_path, detection, crop, apply_rotation = payload
    if detection.translation_x is None or detection.translation_y is None:
        return False
    render_frame(input_path, output_path, detection, crop=crop, apply_rotation=apply_rotation)
    return True


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

    document = metadata_document(
        args.input,
        detections,
        crop=None,
    )
    document["common_radius"] = common_radius
    document["rotation_boundaries"] = [boundary.__dict__ for boundary in rotation_boundaries]
    write_metadata(args.metadata, document)

    if args.previews:
        write_previews(inputs, detections, Path(args.previews), config, args.preview_max_dim, jobs)

    print(f"Detected {len(detections)} frames: {summarize(detections)}")
    if common_radius is not None:
        print(f"Common radius estimate: {common_radius:.2f}px")
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
    skipped = 0
    payloads = []
    for path in inputs:
        detection = by_name.get(path.name)
        if detection is None or detection.translation_x is None or detection.translation_y is None:
            skipped += 1
            continue
        payloads.append((str(path), str(output_dir / path.name), detection, crop, apply_rotation))
    rendered = sum(
        1
        for result in parallel_map_ordered(render_worker, payloads, jobs=jobs, desc="render")
        if result
    )
    print(f"Rendered {rendered} frames to {output_dir}")
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
    crop = resolve_crop(args, detections)
    crop_data = crop.to_dict() if crop is not None else None
    document = metadata_document(args.input, detections, crop=crop_data)
    document["common_radius"] = common_radius
    document["rotation_boundaries"] = [boundary.__dict__ for boundary in rotation_boundaries]
    write_metadata(metadata_path, document)

    write_previews(inputs, detections, previews_dir, config, args.preview_max_dim, jobs)

    output_dir = Path(args.output)
    payloads = [
        (str(path), str(output_dir / path.name), detection, crop, True)
        for path, detection in zip(inputs, detections)
        if detection.translation_x is not None and detection.translation_y is not None
    ]
    rendered = sum(
        1
        for result in parallel_map_ordered(render_worker, payloads, jobs=jobs, desc="render")
        if result
    )

    print(f"Detected {len(detections)} frames: {summarize(detections)}")
    print(f"Rendered {rendered} frames to {output_dir}")
    print(f"Metadata: {metadata_path}")
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
    parallel_map_ordered(preview_worker, payloads, jobs=jobs, desc="previews")


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
