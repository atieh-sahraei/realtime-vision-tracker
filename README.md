# Robust Real-Time Vision Tracker

A small real-time computer-vision project focused on **system reliability around a detector** rather than on detector architecture alone.

The project combines:

- image processing
- object detection
- Kalman tracking
- temporal prediction
- occlusion handling
- confidence-based state logic
- real-time performance monitoring
- CSV logging
- modular, testable software

It is intentionally a **generic object-tracking system**, not a hand-tracking or pose-estimation product.

## Why this project

A vision model can give good frame-by-frame predictions and still produce an unstable product.

Real systems often need additional logic around the detector:

```text
Camera / Video
      |
      v
   Detector
      |
      v
Measurement confidence
      |
      v
Kalman prediction + correction
      |
      v
Reliability / state logic
      |
      v
Stable system output
```

This project demonstrates that system-level layer.

## Demo behavior

The built-in synthetic scene contains:

- a moving target
- background distractors
- image noise
- two temporary occlusions

During normal visibility, the target is detected and tracked.

During temporary occlusion, the detector loses the target but the Kalman filter continues predicting its motion.

A state machine changes the system state between:

- `TRACKING`
- `UNCERTAIN`
- `LOST`

rather than blindly trusting every frame.

## Project structure

```text
robust_realtime_vision_tracker/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── main.py
│   ├── detector.py
│   ├── kalman_tracker.py
│   ├── state_logic.py
│   └── demo_scene.py
└── tests/
    └── test_state_logic.py
```

## Installation

```bash
python -m venv .venv
```

Activate the environment and install:

```bash
pip install -r requirements.txt
```

## Run the built-in demo

```bash
python src/main.py
```

Press `q` or `Esc` to stop.

The synthetic demo is useful because no model weights or test video are required.

## Run with a video

```bash
python src/main.py --video path/to/video.mp4
```

## Run with a webcam

```bash
python src/main.py --camera 0
```

## Optional neural-network detector

The default demo uses a classical color detector so the project runs immediately without downloading weights.

The tracking and reliability system can also wrap a YOLO-style ONNX model through OpenCV DNN:

```bash
python src/main.py \
  --video path/to/video.mp4 \
  --model path/to/model.onnx
```

To track only one class:

```bash
python src/main.py \
  --video path/to/video.mp4 \
  --model path/to/model.onnx \
  --class-id 0
```

Model weights are intentionally not committed to this repository.

The ONNX adapter and classical detector produce the same `Detection` structure, so the downstream Kalman tracker, confidence logic, occlusion handling, logging, and performance monitoring do not need to change.

This separation is deliberate:

```text
Classical detector ----\
                        >---- Common Detection ----> Tracking / Reliability
ONNX neural detector --/
```

## CSV logging

Each run writes a frame-by-frame CSV containing:

- detection availability
- detection confidence
- measured position
- filtered/tracked position
- estimated velocity
- tracking state
- reliability score
- processing time
- smoothed FPS

Default:

```text
outputs/tracking_log.csv
```

## Main design choices

### Detector abstraction

The included detector is intentionally simple: a green-object detector using HSV thresholding and morphology.

The tracker and state logic do **not** depend on that implementation.

That means the detector can later be replaced by:

- an ONNX object detector
- a YOLO model
- a pose estimator
- a segmentation model
- another application-specific detector

without redesigning the tracking and reliability logic.

### Kalman filtering

The tracker uses a constant-velocity state:

```text
[x, y, vx, vy]
```

This helps smooth noisy detections and predict motion briefly when measurements disappear.

### System-level state logic

The pipeline does not treat every detection equally.

It uses confidence and missing-frame history to provide a more stable state:

```text
TRACKING -> UNCERTAIN -> LOST
```

This is deliberately separate from the detector.

## What this project demonstrates

- OpenCV
- NumPy
- real-time image processing
- tracking
- Kalman filtering
- temporal reasoning
- confidence handling
- occlusion recovery
- algorithmic decision logic
- performance measurement
- data logging
- modular software design
- basic testing

## Limitations

This is a portfolio-sized engineering demo, not a production tracker.

Important limitations:

1. The built-in demo detector is classical image processing; an optional YOLO-style ONNX adapter is included, but model weights are not bundled.
2. It tracks only one target.
3. The Kalman model assumes approximately constant velocity.
4. It does not yet use appearance features for re-identification.
5. Camera calibration and 3D geometry are outside the current scope.

These limitations are intentional and leave clear paths for future development.

## Useful next improvements

- replace the classical detector with an ONNX neural-network detector
- add detector/tracker association for multiple objects
- add optical flow as a secondary motion cue
- add camera calibration and pixel-to-world geometry
- add a benchmark script for latency and tracking error
- add Docker support
- add GitHub Actions tests
- add unit tests for Kalman behavior
