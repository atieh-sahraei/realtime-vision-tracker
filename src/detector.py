from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    confidence: float
    area: float


class ColorBlobDetector:
    """
    Simple detector used to keep the demo self-contained.

    It detects a bright green object in HSV space. The rest of the project is
    intentionally independent of the detector, so this class can later be
    replaced by a neural-network detector without changing the tracker or
    decision logic.
    """

    def __init__(
        self,
        lower_hsv=(35, 80, 80),
        upper_hsv=(90, 255, 255),
        min_area: float = 250.0,
    ):
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.min_area = float(min_area)

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))

        if area < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w / 2.0, y + h / 2.0)

        confidence = float(np.clip(area / 4000.0, 0.0, 1.0))

        return Detection(
            center=center,
            bbox=(x, y, w, h),
            confidence=confidence,
            area=area,
        )
