# Solar Eclipse Timelapse Aligner

Command-line tools for aligning OpenEXR solar eclipse timelapse frames. The
pipeline detects the eclipse center in each frame, writes detection metadata and
diagnostic previews, and renders aligned EXR masters for later grading or video
assembly.

The tool preserves the original input frames. Frames with clipped or uncertain
detections are still tracked in metadata and diagnostics so they can be reviewed.

## Requirements

On Ubuntu or Linux Mint, install the expected system packages:

```bash
sudo apt install python3-numpy python3-opencv python3-openimageio openimageio-tools openexr python3-pytest python3-tqdm
```

Run commands from this directory with Python 3.12 or newer.

## Recommended Commands

Process the current Darktable EXR export in one pass:

```bash
python3 -m eclipse_align process \
  --input "1-200/darktable_exported/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --jobs 4
```

Use the two-step workflow when tuning detection settings or inspecting metadata:

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

Enable roll correction across manual reframe segments when needed:

```bash
python3 -m eclipse_align process \
  --input "1-200/darktable_exported/*.exr" \
  --output aligned \
  --diagnostics diagnostics \
  --crop \
  --margin 80 \
  --detect-rotation \
  --jobs 4
```

Run the tests:

```bash
pytest
```
