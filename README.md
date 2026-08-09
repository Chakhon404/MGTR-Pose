# MG-Pose: Hybrid Adaptive Skipping System

A refactored, modular Human Pose Estimation system using YOLO + Optical Flow for adaptive inference skipping.

## Project Structure

```
re_MG-Pose/
├── src/
│   ├── config.py              # Configuration and constants
│   ├── main.py                 # Main entry point
│   ├── utils/                  # Utility functions
│   │   ├── math_utils.py      # Interpolation, keypoint math
│   │   ├── video_utils.py     # Video I/O helpers
│   │   └── visualization.py   # Skeleton drawing
│   ├── core/                   # Core algorithms
│   │   ├── yolo_wrapper.py    # YOLO model wrapper
│   │   └── optical_flow.py    # Sparse & Dense flow
│   ├── pipeline/               # Processing pipelines
│   │   ├── base.py            # Base pipeline class
│   │   ├── pure_yolo.py       # Pure YOLO
│   │   ├── hybrid_sparse.py   # Hybrid Sparse Flow
│   │   └── hybrid_dense.py    # Hybrid Dense Flow
│   ├── evaluation/            # Evaluation tools
│   │   ├── plots_result_fps.py
│   │   ├── plot_result_mpjpe.py
│   │   └── plot_jitter.py
│   ├── live/                  # Real-time webcam
│   │   ├── pure_live.py
│   │   └── hybrid_live.py
│   └── legacy/                # Legacy 5-fold scripts
├── models/                     # YOLO models
├── data/
│   ├── inputs/                 # Input videos
│   └── outputs/               # Output NPZ files
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Download YOLO pose models to `models/`:
- yolov8n-pose.pt
- yolov8m-pose.pt

## Usage

### Run pipelines

```bash
# Pure YOLO
python -m src.main --pipeline pure --video test.mp4

# Hybrid Sparse Flow (recommended)
python -m src.main --pipeline sparse --video test.mp4 --motion_thr 3.0 --max_skip 6

# Hybrid Dense Flow
python -m src.main --pipeline dense --video test.mp4 --motion_thr 0.9 --max_skip 2
```

### Live Webcam

```bash
# Pure YOLO live
python -m src.live.pure_live --camera 1

# Hybrid live with buffer
python -m src.live.hybrid_live --camera 1 --motion_thr 10.0
```

### Evaluation

```bash
# MPJPE evaluation
python -m src.evaluation.plot_result_mpjpe --pred_dir ./pred --gt_dir ./gt

# Jitter analysis
python -m src.evaluation.plot_jitter --input_dir ./results

# FPS comparison
python -m src.evaluation.plots_result_fps --input_dir ./results
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--pipeline` | Pipeline type: pure/sparse/dense | sparse |
| `--video` | Input video file | Idle.mp4 |
| `--model` | YOLO model | yolov8n-pose.pt |
| `--conf` | Confidence threshold | 0.25 |
| `--device` | Device: cpu or 0 | cpu |
| `--motion_thr` | Motion threshold | 3.0 |
| `--max_skip` | Max frames to skip | 6 |
| `--flow_scale` | Flow resize factor | 0.25 |

## Architecture

- **Pure YOLO**: Run YOLO on every frame (baseline)
- **Hybrid Sparse**: Use Lucas-Kanade optical flow to detect motion, skip YOLO when idle
- **Hybrid Dense**: Use Farneback dense flow for global motion detection

The 2-gears adaptive system detects motion score and decides whether to run YOLO (keyframe) or skip (interpolate).
