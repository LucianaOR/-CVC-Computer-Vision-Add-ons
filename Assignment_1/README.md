# Computer Vision Add-ons — Assignment 1

## Overview

This project explores local feature detection, description, and matching techniques for image correspondence, using the HPatches dataset.

The implementation includes classical computer vision methods and an experimental pipeline for evaluating matching performance under illumination and viewpoint changes.

## Project Structure

- `src/` — Feature extraction, matching, evaluation, and visualization
- `main.py` — Classical methods evaluation
- `main_learned.py` — SuperPoint + LightGlue evaluation (experimental)
- `results/` — Generated figures and evaluation tables
- `hpatches-sequences-release/` — HPatches dataset

## Execution

Install the required Python dependencies and run:

```bash
python main.py
```

For the experimental learned-feature pipeline, after installing LightGlue:

```bash
python main_learned.py
```

## Current Status

The classical feature evaluation pipeline has been implemented and tested on selected HPatches sequences.

The SuperPoint + LightGlue integration is experimental and has not yet been fully validated. Additional experiments, comparisons, and refinements remain to be completed.