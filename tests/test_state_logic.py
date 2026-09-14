import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_logic import ReliabilityStateMachine, TrackingState


def test_tracking_when_confidence_is_high():
    sm = ReliabilityStateMachine()
    result = sm.update(0.8)
    assert result.state == TrackingState.TRACKING


def test_uncertain_when_detection_missing():
    sm = ReliabilityStateMachine()
    result = sm.update(None)
    assert result.state == TrackingState.UNCERTAIN


def test_lost_after_many_missing_frames():
    sm = ReliabilityStateMachine(max_missing_before_lost=3)

    sm.update(0.8)
    sm.update(None)
    sm.update(None)
    sm.update(None)
    result = sm.update(None)

    assert result.state == TrackingState.LOST
