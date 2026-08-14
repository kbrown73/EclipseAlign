# Solar Eclipse Timelapse Aligner

Command-line tools for aligning OpenEXR solar eclipse timelapse frames. The
pipeline detects the eclipse center in each frame, writes detection metadata and
diagnostic previews, and renders aligned EXR masters for later grading or video
assembly.

The tool preserves the original input frames. Frames with clipped or uncertain
detections are still tracked in metadata and diagnostics so they can be reviewed.
When rotation detection is enabled, `diagnostics/rotation_summary.csv` records
one graphable row per reframe boundary.

## Requirements

On Ubuntu or Linux Mint, install the expected system packages:

```bash
sudo apt install python3-numpy python3-opencv python3-openimageio openimageio-tools openexr python3-pytest python3-tqdm
```

Run commands from this directory with Python 3.12 or newer.

## Recommended Commands

Process the current Darktable EXR export in one pass:

```bash
/usr/bin/python3 -m eclipse_align process \
  --input "1-200/darktable_exported/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --jobs 4
```

Use the two-step workflow when tuning detection settings or inspecting metadata:

```bash
/usr/bin/python3 -m eclipse_align detect \
  --input "1-200/darktable_exported/*.exr" \
  --metadata diagnostics/detections.json \
  --previews diagnostics/previews \
  --jobs 4

/usr/bin/python3 -m eclipse_align render \
  --input "1-200/darktable_exported/*.exr" \
  --metadata diagnostics/detections.json \
  --output aligned \
  --crop \
  --margin 80 \
  --jobs 4
```

Add `--reformat-output` to `render` or `process` to write continuous sequence
names like `frame_0001.exr`, `frame_0002.exr`, starting from the first rendered
frame.

Add `--alpha-circle` to `render` or `process` to add the fitted solar disk as a
filled white alpha channel in each rendered EXR.

Enable conservative roll correction across manual reframe segments when needed.
This estimates the drift direction within each reframe segment and corrects
abrupt drift-angle changes; a texture-focused registration pass checks the
frames around each reframe boundary and can refine those corrections:

```bash
/usr/bin/python3 -m eclipse_align process \
  --input "1-200/darktable_exported/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --detect-rotation \
  --jobs 4
```

Optionally apply a bounded post-render residual translation polish pass:

```bash
/usr/bin/python3 -m eclipse_align process \
  --input "1-200/darktable_exported/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --detect-rotation \
  --polish-alignment \
  --jobs 4
```

Write sensor-fixed dust candidate diagnostics without changing rendered pixels:

```bash
/usr/bin/python3 -m eclipse_align process \
  --input "1-200/darktable_exported/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --detect-dust \
  --jobs 4
```

Dust diagnostics are written to `diagnostics/dust/`:

- `dust_summary.csv`: candidate centers, approximate radii, and support scores.
- `dust_score.png`: heat map of repeated dark local-contrast hits in camera coordinates.
- `dust_mask.png`: area-filtered candidate mask.
- `candidates/*.png`: per-frame candidate overlays for visual inspection.

The default dust pass now inspects out to `1.03x` the fitted solar radius so
spots near the limb are included. To push detection further while reviewing
diagnostics, lower the repeated-hit threshold:

```bash
/usr/bin/python3 -m eclipse_align detect \
  --input "1-200/darktable_exported/*.exr" \
  --metadata diagnostics/detections.json \
  --detect-dust \
  --dust-min-hit-fraction 0.08 \
  --dust-min-deficit 0.025 \
  --previews diagnostics/previews \
  --jobs 4
```

Run the tests:

```bash
pytest
```
