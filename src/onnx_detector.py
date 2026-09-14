from typing import Optional

import cv2
import numpy as np

from detector import Detection


class YoloOnnxDetector:
    """Minimal YOLO-style ONNX detector using OpenCV DNN."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        class_id: Optional[int] = None,
    ):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.class_id = class_id

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        output = self.net.forward()
        predictions = np.squeeze(output)

        if predictions.ndim != 2:
            raise RuntimeError(f"Unexpected ONNX output shape: {output.shape}")

        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes = []
        confidences = []

        x_scale = width / float(self.input_size)
        y_scale = height / float(self.input_size)

        for row in predictions:
            if row.shape[0] < 5:
                continue

            cx, cy, w, h = row[:4]
            class_scores = row[4:]
            if class_scores.size == 0:
                continue

            candidate_class = int(np.argmax(class_scores))
            confidence = float(class_scores[candidate_class])

            if confidence < self.confidence_threshold:
                continue
            if self.class_id is not None and candidate_class != self.class_id:
                continue

            x = int((cx - 0.5 * w) * x_scale)
            y = int((cy - 0.5 * h) * y_scale)
            bw = int(w * x_scale)
            bh = int(h * y_scale)

            boxes.append([x, y, bw, bh])
            confidences.append(confidence)

        if not boxes:
            return None

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return None

        retained = [int(i) for i in np.array(indices).flatten()]
        best = max(retained, key=lambda i: confidences[i])

        x, y, w, h = boxes[best]
        center = (x + w / 2.0, y + h / 2.0)

        return Detection(
            center=center,
            bbox=(x, y, w, h),
            confidence=confidences[best],
            area=float(max(0, w) * max(0, h)),
        )
