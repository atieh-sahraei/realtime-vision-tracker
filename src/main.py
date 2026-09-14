import argparse
import csv
import time
from pathlib import Path

import cv2

from detector import ColorBlobDetector
from demo_scene import SyntheticScene
from kalman_tracker import KalmanPointTracker
from onnx_detector import YoloOnnxDetector
from state_logic import ReliabilityStateMachine, TrackingState


def draw_label(frame, text, xy, color=(255, 255, 255), scale=0.6):
    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def state_color(state):
    if state == TrackingState.TRACKING:
        return (0, 220, 0)
    if state == TrackingState.UNCERTAIN:
        return (0, 200, 255)
    if state == TrackingState.LOST:
        return (0, 0, 255)
    return (180, 180, 180)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Robust real-time object tracking demo."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to video. If omitted, synthetic demo is used.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Optional webcam index.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/tracking_log.csv",
        help="CSV log path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional YOLO-style ONNX model path.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Optional class ID to track when using --model.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without GUI windows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model:
        detector = YoloOnnxDetector(args.model, class_id=args.class_id)
        print(f"Using ONNX detector: {args.model}")
    else:
        detector = ColorBlobDetector()
        print("Using built-in color detector")

    tracker = KalmanPointTracker()
    state_machine = ReliabilityStateMachine()

    source_mode = "demo"
    cap = None
    scene = None

    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_mode = "video"
    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        source_mode = "camera"
    else:
        scene = SyntheticScene()

    if cap is not None and not cap.isOpened():
        raise RuntimeError("Could not open video/camera source.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    fps_ema = None
    alpha = 0.10

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame",
                "timestamp_s",
                "detected",
                "det_confidence",
                "measured_x",
                "measured_y",
                "tracked_x",
                "tracked_y",
                "velocity_x",
                "velocity_y",
                "state",
                "reliability",
                "processing_ms",
                "fps_smoothed",
            ]
        )

        start_time = time.perf_counter()

        while True:
            frame_start = time.perf_counter()

            if source_mode == "demo":
                if frame_index >= scene.total_frames:
                    break
                frame = scene.frame(frame_index)
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            detection = detector.detect(frame)
            measurement = detection.center if detection is not None else None
            track = tracker.update(measurement)

            confidence = detection.confidence if detection is not None else None
            state = state_machine.update(confidence)

            measured_x = ""
            measured_y = ""
            if detection is not None:
                measured_x, measured_y = detection.center
                x, y, w, h = detection.bbox
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 180, 0),
                    2,
                )
                cv2.circle(
                    frame,
                    (int(measured_x), int(measured_y)),
                    5,
                    (255, 180, 0),
                    -1,
                )

            tracked_x = tracked_y = velocity_x = velocity_y = ""
            if track is not None:
                tracked_x, tracked_y = track.position
                velocity_x, velocity_y = track.velocity

                color = state_color(state.state)

                cv2.circle(
                    frame,
                    (int(tracked_x), int(tracked_y)),
                    8,
                    color,
                    2,
                )

                prediction_end = (
                    int(tracked_x + 8.0 * velocity_x),
                    int(tracked_y + 8.0 * velocity_y),
                )
                cv2.arrowedLine(
                    frame,
                    (int(tracked_x), int(tracked_y)),
                    prediction_end,
                    color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.25,
                )

            processing_ms = (time.perf_counter() - frame_start) * 1000.0
            fps_now = 1000.0 / processing_ms if processing_ms > 0 else 0.0

            if fps_ema is None:
                fps_ema = fps_now
            else:
                fps_ema = alpha * fps_now + (1 - alpha) * fps_ema

            timestamp_s = time.perf_counter() - start_time

            draw_label(
                frame,
                f"State: {state.state.value}",
                (20, 35),
                state_color(state.state),
                0.75,
            )
            draw_label(frame, f"Reliability: {state.reliability:.2f}", (20, 65))
            draw_label(frame, f"Processing: {processing_ms:.2f} ms", (20, 95))
            draw_label(frame, f"FPS (EMA): {fps_ema:.1f}", (20, 125))

            writer.writerow(
                [
                    frame_index,
                    f"{timestamp_s:.6f}",
                    int(detection is not None),
                    "" if confidence is None else f"{confidence:.4f}",
                    measured_x,
                    measured_y,
                    tracked_x,
                    tracked_y,
                    velocity_x,
                    velocity_y,
                    state.state.value,
                    f"{state.reliability:.4f}",
                    f"{processing_ms:.4f}",
                    f"{fps_ema:.4f}",
                ]
            )

            if not args.no_display:
                cv2.imshow("Robust Real-Time Vision Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_index} frames")
    print(f"CSV log saved to: {output_path}")


if __name__ == "__main__":
    main()
