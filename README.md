# Hassy Film Slice

A Python tool for automatically splitting (Hasselblad X1/X5) scanned film strips into individual frames.

## Background

When scanning film with scanners like the Hasselblad X1/X5, entire film strips need to be fed into the scanner, for example:
- 120 format 6x6 film scans result in strips containing 3 frames
- 135 format film scans result in strips containing 6x2 (12) frames

This tool automatically detects and splits these scanned strips into individual frame images, preserving the original image quality and metadata.

## Features

- Automatic detection of frame boundaries in scanned film strips
- Support for both horizontal and vertical film strips
- Handles both 8-bit and 16-bit TIFF/JPG/PNG images
- Preserves original image quality
- Batch processing support for multiple files and directories

## Installation

### Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy

```bash
pip install numpy opencv-python
```

## Usage

### Basic Usage

```bash
python hassy_film_slicer.py input_file.tif
```

### Process an Entire Directory

```bash
python hassy_film_slicer.py input_directory
```

### Advanced Options

```bash
python hassy_film_slicer.py input_file.tif --white_threshold 240 --black_threshold 15
```

#### Parameters

- `white_threshold`: Value between 0-255 (for both 8-bit / 16-bit). Pixels with values above this threshold are considered white. Default is 240 (for 8-bit) or scaled for 16-bit images.
- `black_threshold`: Value between 0-255 (for both 8-bit / 16-bit). Pixels with values below this threshold are considered black. Default is 15 (for 8-bit) or scaled for 16-bit images.

Adjusting these thresholds can help improve frame detection in scans with different exposure levels or contrast.

## How It Works

1. The tool loads the scanned image
2. Detects frame boundaries using projection analysis:
   - Converts the image to grayscale
   - Applies threshold processing to identify black separation lines
   - Analyzes horizontal and vertical projections to locate frame boundaries
3. Splits the original image into individual frames
4. Saves each frame as a separate TIFF file, preserving original quality and metadata

## Output

Output files are named using the pattern: `{original_filename}_frame_{number}.tif`
