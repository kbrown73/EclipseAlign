# Solar Eclipse Timelapse Aligner

Command-line tools for aligning OpenEXR solar eclipse timelapse frames.

The pipeline detects the eclipse center in each frame, writes metadata and
diagnostic previews, then renders aligned EXR masters for later grading or video
assembly. The original input frames are never modified.

Frames with clipped or uncertain detections are still recorded in the metadata
and diagnostics so they can be reviewed before rendering.

## Requirements

Run commands from this directory with Python 3.12 or newer.

On Ubuntu or Linux Mint, install the expected packages with:

```bash
sudo apt install python3-numpy python3-opencv python3-openimageio openimageio-tools openexr python3-pytest python3-tqdm
```

Input frames are expected to be `.exr` files. Video files can be decoded to an
intermediate EXR sequence with `extract-video`.

## Quick Start

For the usual one-pass workflow, detect the eclipse, write diagnostics, crop the
result, and render aligned frames:

```bash
/usr/bin/python3 -m eclipse_align process \
  --input "path/to/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --jobs 4
```

This writes aligned EXR frames to `aligned/` and diagnostics to `diagnostics/`.

## Video Input

Video input is currently handled as an explicit extraction step. Decode the
video to EXR frames first, then run the normal EXR workflow:

```bash
/usr/bin/python3 -m eclipse_align extract-video \
  --input "path/to/eclipse.mp4" \
  --output extracted_frames

/usr/bin/python3 -m eclipse_align process \
  --input "extracted_frames/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --jobs 4
```

For one event split across multiple video files, pass the files in the order you
want them decoded. `--input` may be repeated, or one `--input` may be followed
by multiple paths:

```bash
/usr/bin/python3 -m eclipse_align extract-video \
  --input "path/to/video1.mp4" \
  --input "path/to/video2.mp4" \
  --output extracted_frames

/usr/bin/python3 -m eclipse_align extract-video \
  --input "path/to/video1.mp4" "path/to/video2.mp4" \
  --output extracted_frames
```

By default, AstroIO chooses the decoded video precision automatically: 8-bit
sources are decoded as `rgb24`, while higher bit-depth sources are decoded as
`rgb48le`. The extractor scales integer video values to floating `0..1` EXR
values and applies `--transfer srgb` by default so normal display-referred video
is written as linear EXR data. Use `--transfer none` to write scaled but
non-linear encoded RGB values.

The extractor writes FLOAT EXRs for decoded `uint16` frames to avoid losing
precision, and HALF EXRs otherwise. You can override this with
`--output-format` or `--exr-pixel-type`.

## Two-Step Workflow

Use `detect` and `render` separately when tuning detection settings, inspecting
metadata, or reviewing previews before rendering the final frames:

```bash
/usr/bin/python3 -m eclipse_align detect \
  --input "path/to/*.exr" \
  --metadata diagnostics/detections.json \
  --previews diagnostics/previews \
  --jobs 4

/usr/bin/python3 -m eclipse_align render \
  --input "path/to/*.exr" \
  --metadata diagnostics/detections.json \
  --output aligned \
  --crop \
  --margin 80 \
  --jobs 4
```

## Useful Options

- Add `--reformat-output` to `render` or `process` to write continuous sequence
  names such as `frame_0001.exr`, `frame_0002.exr`, starting from the first
  rendered frame.
- Add `--alpha-circle` to `render` or `process` to add the fitted solar disk as
  a filled white alpha channel in each rendered EXR.
- Use `--threshold` to tune where the normalized bright-limb mask is traced.
  Lower values fit a larger disk on soft-limb frames; the default is `0.08`.
- Add `--prefer-plausible-raw` to `detect` or `process` only after reviewing
  diagnostics when the raw pre-refinement fit is better than interpolation in
  obstructed frames. Use `--plausible-raw-range 1225-1300,2000-2100,3720+`
  to limit that behavior to reviewed 1-based frame ranges. A suffix like
  `3720+` means frame 3720 through the final input frame.
- Add `--horizon-ellipse-range 1225-1300,2000-2100,3720+` to `detect` or `process`
  for reviewed sunset/horizon ranges where the apparent solar disk is visibly
  flattened. Accepted fits are gated by ellipse residual, support, tilt, radius,
  and center drift, then recorded with `horizon_ellipse_fit`.

## Rotation Correction

Enable conservative roll correction when the source sequence includes manual
reframe segments. The tool estimates the drift direction within each segment,
corrects abrupt drift-angle changes, and uses texture-focused registration
around each reframe boundary to refine the correction.

```bash
/usr/bin/python3 -m eclipse_align process \
  --input "path/to/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --detect-rotation \
  --jobs 4
```

When rotation detection is enabled, `diagnostics/rotation_summary.csv` records
one graphable row per reframe boundary.

## Alignment Polish

After rendering, you can apply a bounded residual translation polish pass:

```bash
/usr/bin/python3 -m eclipse_align process \
  --input "path/to/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --detect-rotation \
  --polish-alignment \
  --jobs 4
```

## Dust Diagnostics

Write sensor-fixed dust candidate diagnostics without changing rendered pixels:

```bash
/usr/bin/python3 -m eclipse_align process \
  --input "path/to/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --detect-dust \
  --jobs 4
```

Dust diagnostics are written to `diagnostics/dust/`:

- `dust_summary.csv`: candidate centers, approximate radii, and support scores.
- `dust_score.png`: heat map of repeated dark local-contrast hits in camera
  coordinates.
- `dust_mask.png`: area-filtered candidate mask.
- `candidates/*.png`: per-frame candidate overlays for visual inspection.

The default dust pass now inspects out to `1.03x` the fitted solar radius so
spots near the limb are included. To push detection further while reviewing
diagnostics, lower the repeated-hit threshold:

```bash
/usr/bin/python3 -m eclipse_align detect \
  --input "path/to/*.exr" \
  --metadata diagnostics/detections.json \
  --detect-dust \
  --dust-min-hit-fraction 0.08 \
  --dust-min-deficit 0.025 \
  --previews diagnostics/previews \
  --jobs 4
```

## Tests

Run the tests:

```bash
pytest
```
