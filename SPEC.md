# Solar Eclipse Timelapse Aligner V1 Spec

## Context

This repo contains source frames for a 2026 solar eclipse timelapse. The current processing input is a sequence of OpenEXR files exported from Darktable:

- Input directory: `1-200/darktable_exported`
- Input format: `.exr`
- Approximate frame count observed during spec drafting: 716
- Example frame geometry observed during spec drafting: `5494x3666`

The timelapse was shot from a static tripod with occasional manual reframing every 8-10 minutes. The sun/eclipsed sun therefore drifts through the frame and then jumps after each reframing. Some frames include the eclipse partially out of frame because a manual correction was made late.

The first version of the tool should produce centered, aligned EXR frames for later processing in other software. It should preserve the original frames.

## Goals

- Detect the eclipse/sun location in each EXR frame.
- Align every frame so the detected eclipse center is placed at the image center.
- Keep output masters as EXR.
- Attempt to align partially clipped frames where enough visible limb data remains.
- Flag clipped, low-confidence, or failed detections for review.
- Optionally crop aligned frames to remove unused borders.
- Generate diagnostics that make detection problems easy to inspect.

## Non-Goals For V1

- Reconstruct missing image data from clipped frames.
- Stabilize rotation or lunar contact angle.
- Build a GUI.
- Render the final video.
- Perform creative color grading or tone mapping.
- Modify or overwrite original inputs.

## Important Constraints

If the eclipse is clipped in the source image, the tool can estimate the true disk center and align the visible content, but it cannot recover missing pixels. These frames should remain available in the output but must be marked in metadata and diagnostics.

The detection should not rely on a fixed absolute brightness threshold because the sequence changes brightness significantly as the sun sets.

The detection should be robust against transient foreground objects such as aircraft, birds, or dust-like artifacts crossing the frame.

## Dependencies

Recommended Ubuntu/Linux Mint packages:

```bash
sudo apt install python3-numpy python3-opencv python3-openimageio openimageio-tools openexr python3-pytest python3-tqdm
```

Primary expected package usage:

- `numpy`: image arrays and numeric fitting.
- `cv2` from `python3-opencv`: masks, morphology, contours, transforms, preview drawing.
- `OpenImageIO`: EXR read/write.
- `tqdm`: progress reporting.
- `pytest`: tests.
- `oiiotool` / OpenEXR CLI tools: manual inspection and debugging.

## V1 Command Shape

Single-command workflow:

```bash
python3 -m eclipse_align process \
  --input "1-200/darktable_exported/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --jobs 4
```

Two-step workflow, useful while tuning:

```bash
python3 -m eclipse_align detect \
  --input "1-200/darktable_exported/*.exr" \
  --metadata diagnostics/detections.json \
  --previews diagnostics/previews \
  --jobs 4

python3 -m eclipse_align render \
  --input "1-200/darktable_exported/*.exr" \
  --metadata diagnostics/detections.json \
  --output aligned \
  --crop \
  --margin 80 \
  --jobs 4
```

## Inputs

- Accept a glob or directory path.
- V1 should support EXR input.
- Sort frames chronologically by filename. For current files, numeric filename order such as `IMG_8480.exr`, `IMG_8482.exr`, etc. is acceptable.
- All frames are expected to have the same dimensions for V1.

## Outputs

Main outputs:

- Aligned EXR frames in the output directory.
- Detection metadata as JSON.
- Diagnostic preview images.

Optional outputs:

- CSV summary for quick spreadsheet inspection.
- Contact sheet of representative diagnostic frames.

Output filenames should preserve the original stem where possible:

```text
aligned/IMG_8480.exr
diagnostics/previews/IMG_8480.png
```

## Detection Pipeline

For each frame:

1. Read EXR as linear RGB.
2. Convert to a floating-point luminance image.
3. Normalize luminance per frame using robust percentiles.
4. Build a bright-object mask using adaptive or percentile-based thresholding.
5. Clean the mask with morphology.
6. Find connected components or contours.
7. Select the most plausible eclipse/sun component by area, shape, location continuity, and expected radius.
8. Extract visible limb/edge points.
9. Fit a circle robustly to the limb points.
10. Allow fitted centers outside the image for clipped frames.
11. Compute confidence and status flags.

### Brightness Handling

Use frame-local statistics rather than fixed thresholds. Initial approach:

- Estimate background and highlight ranges using percentiles.
- Normalize to a stable working range.
- Use a high percentile or Otsu-style threshold on the normalized image.
- Keep CLI knobs for threshold tuning if needed.

### Photobomb Handling

Transient aircraft/birds should not dominate the detection. Initial approach:

- Prefer the largest plausible connected bright component.
- Reject components whose area, radius, or circularity is implausible.
- Use robust circle fitting with outlier rejection.
- Use temporal continuity from neighboring frames to reject sudden one-frame center/radius outliers.

### Clipped Frame Handling

For frames with the eclipse touching or crossing an image edge:

- Mark the frame as `clipped`.
- Fit the circle from the visible limb if enough edge points exist.
- Permit the fitted center to lie outside the image bounds.
- Lower confidence compared with fully visible frames.
- If fitting fails, interpolate or extrapolate center from neighboring valid frames and mark as `estimated`.

## Temporal Tracking

After per-frame detection:

- Estimate a common radius across the sequence from high-confidence frames.
- Smooth centers over time while preserving real jump discontinuities caused by manual reframing.
- Detect outliers where a frame deviates strongly from neighboring motion.
- Fill failed detections from nearby valid detections where possible.

Manual reframing means the observed raw center track is not globally smooth. The aligner should tolerate piecewise smooth motion with occasional jumps.

## Alignment

Default target:

- Image center: `(width / 2, height / 2)`

For each frame:

- Compute translation from detected/estimated center to target center.
- Apply translation to the original EXR image data.
- Fill newly exposed pixels with black.
- Preserve EXR output.

V1 alignment is translation-only.

## Cropping

Supported crop behavior:

- Default: keep full image dimensions with black borders.
- `--crop`: crop to the largest square centered on the alignment target that fits within fully visible `ok` detections.
- `--crop --margin PIXELS`: expand that centered square by the requested margin on all sides. This can intentionally include translated black borders.
- `--manual-crop WxH+X+Y`: explicit crop rectangle escape hatch.
- `--jobs N`: run detection, preview generation, and rendering with N worker processes. `--jobs 1` is the default. `--jobs 0` uses all available CPUs.

The centered square crop is the primary V1 crop behavior. The implementation may still compute internal safe intersections, but the CLI should avoid exposing those implementation details as user-facing modes.

## Metadata

Write JSON metadata with one record per frame:

```json
{
  "filename": "IMG_8480.exr",
  "width": 5494,
  "height": 3666,
  "center_x": 2747.0,
  "center_y": 1833.0,
  "radius": 420.0,
  "confidence": 0.98,
  "status": "ok",
  "flags": [],
  "translation_x": 0.0,
  "translation_y": 0.0
}
```

Possible statuses:

- `ok`
- `clipped`
- `low_confidence`
- `estimated`
- `failed`

Possible flags:

- `touches_left_edge`
- `touches_right_edge`
- `touches_top_edge`
- `touches_bottom_edge`
- `radius_outlier`
- `center_outlier`
- `photobomb_suspected`
- `insufficient_limb`
- `interpolated`
- `distorted_limb_suspected`

Metadata should also include circle-fit quality fields when available:

- `limb_support_fraction`: fraction of detected limb points close to the fitted circle.
- `circle_residual_median_px`: median absolute distance from detected limb points to the fitted circle.
- `circle_residual_p90_px`: 90th percentile absolute distance from detected limb points to the fitted circle.

## Diagnostics

Generate preview images showing:

- Original frame tone-mapped for preview.
- Detected/fitted circle.
- Fitted center.
- Frame status and confidence.
- Optional raw mask preview.

Diagnostic previews should be downscaled by default so a full run does not create unnecessarily large PNGs. Full-resolution previews can be supported as an explicit option.

Diagnostics should make it possible to quickly inspect whether the alignment inputs are trustworthy before rendering all EXRs.

## V2 Rotation Stabilization

V1 translation alignment exposes roll errors introduced during manual reframing. V2 should correct those roll jumps conservatively.

### V2 Rotation Goals

- Detect manual reframe segments from discontinuities in the raw detected eclipse center track.
- Estimate one roll correction per segment, not one free rotation per frame.
- Use image registration around segment boundaries as the primary signal.
- Rotate rendered frames around the aligned eclipse center.
- Store rotation metadata so render can be rerun without redetecting.
- Keep rotation correction opt-in until the diagnostics prove it trustworthy.

### V2 Rotation Non-Goals

- Per-frame rotation jitter correction.
- Full solar/lunar ephemeris modeling.
- Reconstructing clipped-frame data after rotation.
- Depending only on sunspots, which may be sparse, occulted, blurred, or low contrast.

### Rotation Strategy

1. Run the normal detection pass and translation assignment.
2. Detect reframe boundaries from jumps in raw detected centers between adjacent frames.
3. Assign a `segment_id` to every frame.
4. For each boundary, compare one or more frames before and after the boundary:
   - read EXR frames,
   - translate each to the common center,
   - crop a square region around the eclipse,
   - tone-map/normalize the crop,
   - apply a circular/eclipse mask,
   - search a small angle range for the rotation with the best image match.
5. Accumulate boundary rotations into one absolute `rotation_deg` per segment.
6. Store per-frame rotation fields in metadata.
7. During render, apply translation first, then rotate around the target center, then crop.

Primary signal:

- Masked image registration around reframe boundaries.
- This lets the moon shape, sunspots, limb texture, and surface detail all contribute without requiring explicit feature classification.

Secondary/future signals:

- Sunspot feature matching when two or more reliable spots are visible.
- Lunar limb fitting to estimate the vector from solar center to lunar center.
- Manual segment rotation overrides.

### Rotation Metadata

Add these fields per frame:

```json
{
  "segment_id": 0,
  "rotation_deg": 0.0,
  "rotation_confidence": 1.0,
  "rotation_source": "registration"
}
```

Possible `rotation_source` values:

- `none`
- `registration`
- `manual`
- `estimated`

### Rotation CLI

Rotation detection should be opt-in:

```bash
python3 -m eclipse_align detect \
  --input "1-200/darktable_exported/*.exr" \
  --metadata diagnostics/detections.json \
  --previews diagnostics/previews \
  --detect-rotation \
  --jobs 4
```

Rendering should apply rotation automatically when metadata contains non-zero `rotation_deg`, unless disabled:

```bash
python3 -m eclipse_align render \
  --input "1-200/darktable_exported/*.exr" \
  --metadata diagnostics/detections.json \
  --output aligned_rotated \
  --crop \
  --margin 80 \
  --jobs 4
```

Optional render escape hatch:

```bash
python3 -m eclipse_align render \
  --input "1-200/darktable_exported/*.exr" \
  --metadata diagnostics/detections.json \
  --output aligned_translation_only \
  --no-rotation
```

### Rotation Diagnostics

Diagnostics should include:

- Segment boundary list.
- Estimated boundary delta angles.
- Per-segment accumulated rotation.
- Registration confidence/error score.
- Optional before/after boundary preview strips.

### V2 Acceptance Criteria

- Metadata contains stable `segment_id` values for all frames.
- Reframe boundaries correspond to visible manual reframes.
- Rotation correction is constant within each segment.
- Rendering with rotation reduces visible roll jumps at segment boundaries.
- Rendering with `--no-rotation` preserves V1 translation-only behavior.
- Existing V1 detection, crop, and parallel processing tests continue to pass.

## Suggested Repository Structure

```text
eclipse_align/
  __init__.py
  __main__.py
  cli.py
  exr_io.py
  detect.py
  track.py
  render.py
  diagnostics.py
tests/
  test_circle_fit.py
  test_sorting.py
SPEC.md
```

## Implementation Plan

1. Scaffold Python package and CLI.
2. Add EXR read/write using OpenImageIO.
3. Add filename discovery and chronological sorting.
4. Implement luminance conversion and preview tone mapping.
5. Implement bright-object mask and contour extraction.
6. Implement robust circle fitting.
7. Add metadata output.
8. Add diagnostic overlay previews.
9. Add translation-only EXR rendering.
10. Add crop modes.
11. Add tests for sorting, circle fitting, and crop math.

## Acceptance Criteria For V1

- Running `process` on the EXR directory creates aligned EXR outputs.
- Originals remain untouched.
- Metadata is written for every input frame.
- Diagnostic previews are generated.
- Fully visible frames are centered consistently.
- Partially clipped frames are either aligned from visible limb data or marked as estimated/failed.
- Photobomb-like outliers do not permanently perturb the center track.
- The tool exits with a useful summary of ok, clipped, estimated, low-confidence, and failed frames.
