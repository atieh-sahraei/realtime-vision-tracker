from dataclasses import dataclass
from enum import Enum

class TrackingState(str, Enum):
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"

@dataclass
class StateOutput:
    state: TrackingState
    reliability: float

class ReliabilityStateMachine:
    def __init__(self, low_confidence_threshold=0.35, max_missing_before_lost=12):
        self.low_confidence_threshold = low_confidence_threshold
        self.max_missing_before_lost = max_missing_before_lost
        self.missing_frames = 0

    def update(self, detection_confidence):
        if detection_confidence is None:
            self.missing_frames += 1
        else:
            self.missing_frames = 0

        if self.missing_frames > self.max_missing_before_lost:
            return StateOutput(TrackingState.LOST, 0.0)

        if detection_confidence is None:
            reliability = max(0.0, 1.0 - self.missing_frames / self.max_missing_before_lost)
            return StateOutput(TrackingState.UNCERTAIN, reliability)

        if detection_confidence < self.low_confidence_threshold:
            return StateOutput(TrackingState.UNCERTAIN, float(detection_confidence))

        return StateOutput(TrackingState.TRACKING, float(detection_confidence))
