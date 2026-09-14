from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackResult:
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    predicted_only: bool


class KalmanPointTracker:
    """
    Constant-velocity Kalman filter for a 2D point.

    State:
        [x, y, vx, vy]

    Measurement:
        [x, y]
    """

    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False

    def update(
        self, measurement: Optional[Tuple[float, float]]
    ) -> Optional[TrackResult]:
        if not self.initialized:
            if measurement is None:
                return None

            x, y = measurement
            self.kalman.statePost = np.array(
                [[x], [y], [0.0], [0.0]], dtype=np.float32
            )
            self.initialized = True

        prediction = self.kalman.predict()

        predicted_only = measurement is None

        if measurement is not None:
            x, y = measurement
            measured = np.array([[x], [y]], dtype=np.float32)
            corrected = self.kalman.correct(measured)
            state = corrected
        else:
            state = prediction

        return TrackResult(
            position=(float(state[0, 0]), float(state[1, 0])),
            velocity=(float(state[2, 0]), float(state[3, 0])),
            predicted_only=predicted_only,
        )
