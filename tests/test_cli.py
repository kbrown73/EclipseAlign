from argparse import Namespace

from eclipse_align.cli import (
    apply_plausible_raw_overrides,
    build_parser,
    parse_plausible_raw_ranges,
    plausible_raw_indexes,
    resolve_preview_jobs,
)
from eclipse_align.models import FrameDetection


def test_resolve_preview_jobs_leaves_two_cores_free():
    args = Namespace(jobs=8)

    assert resolve_preview_jobs(args) == 6


def test_resolve_preview_jobs_keeps_single_worker_floor():
    args = Namespace(jobs=2)

    assert resolve_preview_jobs(args) == 1


def test_extract_video_input_accepts_repeated_and_multiple_values():
    parser = build_parser()

    args = parser.parse_args(
        [
            "extract-video",
            "--input",
            "first.mp4",
            "--input",
            "second.mp4",
            "third.mp4",
            "--output",
            "frames",
        ]
    )

    assert args.input == [["first.mp4"], ["second.mp4", "third.mp4"]]


def test_plausible_raw_range_parses_repeated_and_comma_values():
    indexes = parse_plausible_raw_ranges(["2-4", "7,9-10"], 12)

    assert indexes == {1, 2, 3, 6, 8, 9}


def test_plausible_raw_range_cli_accepts_comma_separated_range_list():
    parser = build_parser()

    args = parser.parse_args(
        [
            "detect",
            "--input",
            "frames/*.exr",
            "--metadata",
            "diagnostics/detections.json",
            "--plausible-raw-range",
            "1225-1300,3720-4461",
        ]
    )

    assert args.plausible_raw_range == ["1225-1300,3720-4461"]


def test_plausible_raw_range_implies_preference_for_ranges():
    args = Namespace(prefer_plausible_raw=False, plausible_raw_range=["3-5"])

    assert plausible_raw_indexes(args, 10) == {2, 3, 4}


def test_prefer_plausible_raw_selects_all_frames_without_ranges():
    args = Namespace(prefer_plausible_raw=True, plausible_raw_range=[])

    assert plausible_raw_indexes(args, 4) == {0, 1, 2, 3}


def test_apply_plausible_raw_overrides_uses_guarded_raw_fit():
    detections = [
        FrameDetection(
            "a.exr",
            100,
            100,
            center_x=50,
            center_y=50,
            raw_center_x=40,
            raw_center_y=42,
            raw_radius=21,
            radius=20,
            confidence=0.2,
            status="estimated",
            flags=["distorted_limb_suspected", "interpolated"],
            limb_support_fraction=0.4,
            circle_residual_median_px=20,
        )
    ]

    applied = apply_plausible_raw_overrides(detections, {0}, 20)

    assert applied == 1
    assert detections[0].center_x == 40
    assert detections[0].center_y == 42
    assert detections[0].radius == 21
    assert "raw_fit_override" in detections[0].flags
    assert "interpolated" not in detections[0].flags
    assert detections[0].translation_x == 10
    assert detections[0].translation_y == 8


def test_apply_plausible_raw_overrides_rejects_high_residual_fit():
    detections = [
        FrameDetection(
            "a.exr",
            100,
            100,
            center_x=50,
            center_y=50,
            raw_center_x=40,
            raw_center_y=42,
            raw_radius=21,
            radius=20,
            confidence=0.2,
            status="estimated",
            flags=["distorted_limb_suspected", "interpolated"],
            limb_support_fraction=0.4,
            circle_residual_median_px=75,
        )
    ]

    applied = apply_plausible_raw_overrides(detections, {0}, 20)

    assert applied == 0
    assert detections[0].center_x == 50
    assert "raw_fit_override" not in detections[0].flags
