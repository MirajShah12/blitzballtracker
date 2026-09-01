"""
Unit & Integration Tests for Refactored Blitzball Tracking Logic.

Verifies:
1. State-Based Release Gating (STATE_WAITING_RELEASE -> STATE_TRACKING_PITCH only in top 30% of corridor).
2. Strict Area Constraints (rejects < 40 px^2 and > 800 px^2).
3. Forward Kinematics Gating (requires dy > 0 towards plate, local radius r=50px).
4. Stop conditions (reaches strike zone plate plane, 4 consecutive frame misses).
"""

import numpy as np
from tracker import PitchTracker, STATE_WAITING_RELEASE, STATE_TRACKING_PITCH
from detector import BlitzballDetector


def test_state_based_release_gating():
    zone = np.array([[200, 300], [400, 300], [400, 500], [200, 500]], dtype=np.int32)
    tracker = PitchTracker(zone_polygon=zone, conf_thresh=0.25)
    # Set corridor: y from 100 to 500 (height 400). Top 30% is y in [100, 220].
    tracker.set_corridor_box(150, 100, 450, 500)

    assert tracker.state == STATE_WAITING_RELEASE

    # Mock detector to return candidate in lower part of corridor (e.g. batter legs at y=350)
    fake_frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # 1. Candidate at y=350 (outside top 30% release region)
    tracker.detector.detect = lambda frame, roi_box: [(300, 350, 20, 20, 0.85, "blitzball")]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.0)
    assert pt is None, "Candidate outside top 30% of corridor should be rejected in STATE_WAITING_RELEASE"
    assert tracker.state == STATE_WAITING_RELEASE
    assert len(tracker.trajectory) == 0

    # 2. Candidate at y=150 (inside top 30% release region: 100 <= 150 <= 220)
    tracker.detector.detect = lambda frame, roi_box: [(300, 150, 20, 20, 0.85, "blitzball")]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.033)
    assert pt == (300, 150), "Candidate inside top 30% should be accepted"
    assert tracker.state == STATE_TRACKING_PITCH, "Should transition to STATE_TRACKING_PITCH"
    assert len(tracker.trajectory) == 1
    print("✓ test_state_based_release_gating passed")


def test_strict_area_constraints():
    zone = np.array([[200, 300], [400, 300], [400, 500], [200, 500]], dtype=np.int32)
    tracker = PitchTracker(zone_polygon=zone, min_contour_area=40.0, max_contour_area=800.0)
    tracker.set_corridor_box(150, 100, 450, 500)

    fake_frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # 1. Large contour (e.g. 35x35 = 1225 px^2, batter body/leg/bat) in release zone
    tracker.detector.detect = lambda frame, roi_box: [(300, 150, 35, 35, 0.90, "blitzball")]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.0)
    assert pt is None, "Contour area > 800 px^2 must be immediately discarded"
    assert tracker.state == STATE_WAITING_RELEASE

    # 2. Tiny speckle contour (e.g. 5x5 = 25 px^2) in release zone
    tracker.detector.detect = lambda frame, roi_box: [(300, 150, 5, 5, 0.90, "blitzball")]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.0)
    assert pt is None, "Contour area < 40 px^2 must be discarded"
    assert tracker.state == STATE_WAITING_RELEASE

    # 3. Valid ball contour (e.g. 15x15 = 225 px^2)
    tracker.detector.detect = lambda frame, roi_box: [(300, 150, 15, 15, 0.90, "blitzball")]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.0)
    assert pt == (300, 150), "Valid contour area 225 px^2 should be accepted"
    assert tracker.state == STATE_TRACKING_PITCH
    print("✓ test_strict_area_constraints passed")


def test_forward_kinematics_and_local_radius():
    zone = np.array([[200, 300], [400, 300], [400, 500], [200, 500]], dtype=np.int32)
    tracker = PitchTracker(zone_polygon=zone, gate_radius=50.0)
    tracker.set_corridor_box(150, 100, 450, 500)
    fake_frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # Release pitch at (300, 150)
    tracker.detector.detect = lambda frame, roi_box: [(300, 150, 15, 15, 0.90, "blitzball")]
    tracker.process_frame(fake_frame, timestamp=0.0)
    assert tracker.state == STATE_TRACKING_PITCH

    # Frame 2: Candidate with negative forward velocity dy < 0 (moving upward: y=130)
    tracker.detector.detect = lambda frame, roi_box: [(300, 130, 15, 15, 0.90, "blitzball")]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.033)
    assert pt is None, "Candidate with dy <= 0 must be rejected by forward kinematics"
    assert len(tracker.trajectory) == 1

    # Frame 2 retry: Candidate outside local radius (r=50px).
    # Default initial velocity is (0, 15), so predicted position is (300, 165).
    # Candidate at (390, 180): distance hypot(90, 15) = 91.2px > 50px
    tracker.detector.detect = lambda frame, roi_box: [(390, 180, 15, 15, 0.90, "blitzball")]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.033)
    assert pt is None, "Candidate outside local radius r=50px must be rejected"

    # Frame 2 valid: Candidate with positive dy and within r=50px (e.g. at (302, 170))
    # Distance to (300, 165) is hypot(2, 5) = 5.38px <= 50px
    tracker.detector.detect = lambda frame, roi_box: [(302, 170, 15, 15, 0.90, "blitzball")]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.033)
    assert pt == (302, 170), "Valid downward candidate within radius should be accepted"
    assert len(tracker.trajectory) == 2
    print("✓ test_forward_kinematics_and_local_radius passed")


def test_plate_plane_and_miss_termination():
    zone = np.array([[200, 300], [400, 300], [400, 500], [200, 500]], dtype=np.int32)
    # Plate plane is y=500
    tracker = PitchTracker(zone_polygon=zone, gate_radius=50.0, max_consecutive_misses=4)
    tracker.set_corridor_box(150, 100, 450, 500)
    fake_frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # 1. Realistic pitch flight from y=150 towards plate plane at y=500 (step ~ 30px per frame)
    y_coords = [150, 180, 215, 255, 300, 345, 395, 450, 505]
    for i, y in enumerate(y_coords):
        tracker.detector.detect = lambda frame, roi_box, y=y: [(300, y, 15, 15, 0.90, "blitzball")]
        pt, _ = tracker.process_frame(fake_frame, timestamp=i * 0.033)
        assert pt == (300, y), f"Frame {i} at y={y} should be tracked within gate radius"
        if y < 500:
            assert not tracker.is_pitch_complete(), f"Pitch should not complete at y={y} (< 500)"

    assert tracker.is_pitch_complete(), "Pitch must complete when reaching plate plane (y=505 >= 500)"

    eval_res = tracker.evaluate_pitch()
    assert eval_res is not None
    assert eval_res["call"] in ("STRIKE", "BALL")

    # Reset
    tracker.reset()
    assert tracker.state == STATE_WAITING_RELEASE
    assert len(tracker.trajectory) == 0

    # 2. Test 4-consecutive-frame misses termination
    tracker.detector.detect = lambda frame, roi_box: [(300, 150, 15, 15, 0.90, "blitzball")]
    tracker.process_frame(fake_frame, timestamp=0.0)
    tracker.detector.detect = lambda frame, roi_box: [(300, 180, 15, 15, 0.90, "blitzball")]
    tracker.process_frame(fake_frame, timestamp=0.033)

    # 3 misses: not yet complete
    tracker.detector.detect = lambda frame, roi_box: []
    for i in range(3):
        tracker.process_frame(fake_frame, timestamp=0.066 + i * 0.033)
        assert not tracker.is_pitch_complete()

def test_batter_swing_and_movement_rejection():
    """Verify that batter movement, foot repositioning, and bat swings are rejected."""
    zone = np.array([[200, 300], [400, 300], [400, 500], [200, 500]], dtype=np.int32)
    tracker = PitchTracker(zone_polygon=zone, gate_radius=50.0, min_contour_area=40.0, max_contour_area=800.0)
    tracker.set_corridor_box(150, 100, 450, 500)
    fake_frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # 1. Batter wiggles bat at y=320 before pitch is released (in STATE_WAITING_RELEASE)
    tracker.detector.detect = lambda frame, roi_box: [
        (220, 320, 30, 30, 0.90, "blitzball"),  # 900 px^2 (large bat blur)
        (230, 340, 20, 20, 0.85, "blitzball"),  # 400 px^2 (in lower corridor)
    ]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.0)
    assert pt is None
    assert tracker.state == STATE_WAITING_RELEASE

    # 2. Pitch is released at (300, 140)
    tracker.detector.detect = lambda frame, roi_box: [
        (300, 140, 16, 16, 0.92, "blitzball"),  # Valid release ball (256 px^2)
        (230, 340, 20, 20, 0.85, "blitzball"),  # Batter foot at bottom
    ]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.033)
    assert pt == (300, 140)
    assert tracker.state == STATE_TRACKING_PITCH

    # 3. In flight: Batter initiates full swing (large area 40x40=1600 px^2) while ball is at (300, 175)
    tracker.detector.detect = lambda frame, roi_box: [
        (250, 330, 40, 40, 0.95, "blitzball"),  # Bat swing (1600 px^2, should be discarded)
        (300, 175, 16, 16, 0.90, "blitzball"),  # Ball in flight
    ]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.066)
    assert pt == (300, 175), "Must track the ball and discard the bat swing"
    assert len(tracker.trajectory) == 2
    print("✓ test_batter_swing_and_movement_rejection passed")


def test_coasting_and_recovery():
    """Verify coasting behavior for brief occlusion and recovery."""
    zone = np.array([[200, 300], [400, 300], [400, 500], [200, 500]], dtype=np.int32)
    tracker = PitchTracker(zone_polygon=zone, gate_radius=50.0, max_coast_frames=2)
    tracker.set_corridor_box(150, 100, 450, 500)
    fake_frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # Frame 0: Release
    tracker.detector.detect = lambda frame, roi_box: [(300, 150, 15, 15, 0.90, "blitzball")]
    tracker.process_frame(fake_frame, timestamp=0.0)

    # Frame 1: Ball at (300, 180) -> dy = 30
    tracker.detector.detect = lambda frame, roi_box: [(300, 180, 15, 15, 0.90, "blitzball")]
    tracker.process_frame(fake_frame, timestamp=0.033)
    assert len(tracker.trajectory) == 2

    # Frame 2: Occlusion (no detection) -> coasts to (300, 210)
    tracker.detector.detect = lambda frame, roi_box: []
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.066)
    assert pt == (300, 210), f"Expected coasted point (300, 210), got {pt}"
    assert len(tracker.trajectory) == 3

    # Frame 3: Detection recovers at (300, 240)
    tracker.detector.detect = lambda frame, roi_box: [(300, 240, 15, 15, 0.90, "blitzball")]
    pt, _ = tracker.process_frame(fake_frame, timestamp=0.099)
    assert pt == (300, 240)
    assert len(tracker.trajectory) == 4
    print("✓ test_coasting_and_recovery passed")


if __name__ == "__main__":
    test_state_based_release_gating()
    test_strict_area_constraints()
    test_forward_kinematics_and_local_radius()
    test_plate_plane_and_miss_termination()
    test_batter_swing_and_movement_rejection()
    test_coasting_and_recovery()
    print("\nALL TRACKING TESTS PASSED SUCCESSFULLY!")
