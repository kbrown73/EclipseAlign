from argparse import Namespace
from pathlib import Path

from eclipse_align.cli import (
    apply_horizon_ellipse_overrides,
    apply_plausible_raw_overrides,
    build_parser,
    horizon_ellipse_indexes,
    parse_plausible_raw_ranges,
    plausible_raw_indexes,
    resolve_dust_mask_path,
    resolve_preview_jobs,
)
from eclipse_align.detect import DEFAULT_THRESHOLD, DetectionConfig, EllipseFit
from eclipse_align.models import FrameDetection


def test_resolve_preview_jobs_leaves_two_cores_free():
    args = Namespace(jobs=8)

    assert resolve_preview_jobs(args) == 6


def test_resolve_preview_jobs_keeps_single_worker_floor():
    args = Namespace(jobs=2)

    assert resolve_preview_jobs(args) == 1


def test_detection_cli_threshold_defaults_to_detector_default():
    parser = build_parser()

    args = parser.parse_args(
        [
            "detect",
            "--input",
            "frames/*.exr",
            "--metadata",
            "diagnostics/detections.json",
        ]
    )

    assert args.threshold == DEFAULT_THRESHOLD
    assert DetectionConfig().threshold == DEFAULT_THRESHOLD


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
    assert args.debayer == "auto"


def test_extract_video_debayer_accepts_manual_bayer_pattern():
    parser = build_parser()

    args = parser.parse_args(
        [
            "extract-video",
            "--input",
            "raw.avi",
            "--output",
            "frames",
            "--debayer",
            "GRBG",
        ]
    )

    assert args.debayer == "GRBG"


def test_plausible_raw_range_parses_repeated_and_comma_values():
    indexes = parse_plausible_raw_ranges(["2-4", "7,9-10"], 12)

    assert indexes == {1, 2, 3, 6, 8, 9}


def test_plausible_raw_range_parses_open_ended_suffix():
    indexes = parse_plausible_raw_ranges(["2-4,7+"], 10)

    assert indexes == {1, 2, 3, 6, 7, 8, 9}


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
            "1225-1300,2000-2100,3720+",
        ]
    )

    assert args.plausible_raw_range == ["1225-1300,2000-2100,3720+"]


def test_horizon_ellipse_range_cli_accepts_comma_separated_range_list():
    parser = build_parser()

    args = parser.parse_args(
        [
            "detect",
            "--input",
            "frames/*.exr",
            "--metadata",
            "diagnostics/detections.json",
            "--horizon-ellipse-range",
            "1225-1300,2000-2100,3720+",
        ]
    )

    assert args.horizon_ellipse_range == ["1225-1300,2000-2100,3720+"]


def test_render_correct_dust_cli_options():
    parser = build_parser()

    args = parser.parse_args(
        [
            "render",
            "--input",
            "frames/*.exr",
            "--metadata",
            "diagnostics/detections.json",
            "--output",
            "aligned",
            "--correct-dust",
            "--dust-correction-radius",
            "7",
            "--dust-mask-dilation",
            "3",
        ]
    )

    assert args.correct_dust
    assert args.dust_correction_radius == 7
    assert args.dust_mask_dilation == 3


def test_resolve_dust_mask_path_requires_correct_dust(tmp_path):
    mask = tmp_path / "dust_mask.png"
    mask.write_bytes(b"not-an-image-but-existing")
    args = Namespace(correct_dust=False, dust_mask=str(mask), dust_correction_radius=5.0, dust_mask_dilation=2)

    try:
        resolve_dust_mask_path(args, None)
    except SystemExit as exc:
        assert "--dust-mask requires --correct-dust" in str(exc)
    else:
        raise AssertionError("Expected --dust-mask without --correct-dust to fail")


def test_resolve_dust_mask_path_uses_default_when_correcting(tmp_path):
    mask = tmp_path / "dust_mask.png"
    mask.write_bytes(b"not-an-image-but-existing")
    args = Namespace(correct_dust=True, dust_mask=None, dust_correction_radius=5.0, dust_mask_dilation=2)

    assert resolve_dust_mask_path(args, mask) == mask


def test_plausible_raw_range_implies_preference_for_ranges():
    args = Namespace(prefer_plausible_raw=False, plausible_raw_range=["3-5"])

    assert plausible_raw_indexes(args, 10) == {2, 3, 4}


def test_horizon_ellipse_indexes_use_explicit_ranges():
    args = Namespace(horizon_ellipse_range=["2-3,6+"])

    assert horizon_ellipse_indexes(args, 8) == {1, 2, 5, 6, 7}


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


def test_apply_horizon_ellipse_overrides_uses_guarded_fit(monkeypatch):
    detections = [
        FrameDetection(
            "a.exr",
            100,
            100,
            center_x=50,
            center_y=50,
            radius=20,
            confidence=0.25,
            status="estimated",
            flags=["distorted_limb_suspected", "interpolated"],
            limb_support_fraction=0.35,
            circle_residual_median_px=12,
        )
    ]

    ellipse = EllipseFit(
        center_x=47,
        center_y=52,
        major_radius=21,
        minor_radius=17,
        angle_deg=1,
        confidence=0.8,
        residual_median_px=4,
        residual_p90_px=6,
        support_fraction=0.9,
    )

    monkeypatch.setattr("eclipse_align.cli.parallel_map_ordered", lambda fn, items, jobs, desc: [ellipse])

    applied = apply_horizon_ellipse_overrides(
        [Path("a.exr")],
        detections,
        {0},
        DetectionConfig(),
        20,
        jobs=1,
    )

    assert applied == 1
    assert detections[0].center_x == 47
    assert detections[0].center_y == 52
    assert detections[0].radius == 21
    assert detections[0].ellipse_minor_radius == 17
    assert detections[0].translation_x == 3
    assert detections[0].translation_y == -2
    assert "horizon_ellipse_fit" in detections[0].flags
    assert "interpolated" not in detections[0].flags
