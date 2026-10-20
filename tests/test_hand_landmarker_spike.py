from types import SimpleNamespace

from tools.diagnostics import hand_landmarker_spike as spike


detection_rates = spike.detection_rates
fingertip_pixels = spike.fingertip_pixels
ActiveFingertipLock = spike.ActiveFingertipLock
draw_hand_skeletons = spike.draw_hand_skeletons


def test_fingertip_pixels_maps_and_clamps_index_and_middle_tips():
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
    landmarks[8] = SimpleNamespace(x=0.25, y=0.5)
    landmarks[12] = SimpleNamespace(x=1.2, y=-0.2)

    assert fingertip_pixels([landmarks], 200, 100) == [
        (0, 8, 50, 50),
        (0, 12, 199, 0),
    ]


def test_detection_rates_excludes_setup_frames_from_active_window():
    assert detection_rates([False, False, True, True, False, True, False]) == (
        3 / 7,
        3 / 4,
    )


def test_active_fingertip_locks_first_stable_candidate_after_dwell():
    selector = ActiveFingertipLock(dwell_ms=250, radius_px=20)
    left_index = ("Left", 8, 100, 100)
    right_middle = ("Right", 12, 300, 100)

    assert selector.update([left_index, right_middle], 0) is None
    assert selector.update([left_index, right_middle], 249) is None
    assert selector.update([left_index, right_middle], 250) == ("Left", 8)


def test_active_fingertip_ignores_other_hand_until_activity_reset():
    selector = ActiveFingertipLock(dwell_ms=250, radius_px=20)
    left_index = ("Left", 8, 100, 100)
    right_middle = ("Right", 12, 300, 100)

    selector.update([left_index], 0)
    assert selector.update([left_index], 250) == ("Left", 8)
    assert selector.update([right_middle], 500) == ("Left", 8)

    selector.reset()
    assert selector.update([right_middle], 500) is None
    assert selector.update([right_middle], 750) == ("Right", 12)


def test_active_fingertip_movement_restarts_dwell_timer():
    selector = ActiveFingertipLock(dwell_ms=250, radius_px=20)

    assert selector.update([("Left", 8, 100, 100)], 0) is None
    assert selector.update([("Left", 8, 130, 100)], 200) is None
    assert selector.update([("Left", 8, 130, 100)], 449) is None
    assert selector.update([("Left", 8, 130, 100)], 450) == ("Left", 8)


def test_draw_hand_skeletons_uses_all_mediapipe_connections(monkeypatch):
    frame = object()
    hand = [object() for _ in range(21)]
    calls = []
    monkeypatch.setattr(
        spike.mp.tasks.vision.drawing_utils,
        "draw_landmarks",
        lambda *args: calls.append(args),
    )

    draw_hand_skeletons(frame, [hand])

    assert calls == [
        (
            frame,
            hand,
            spike.mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
        )
    ]
    assert len(calls[0][2]) == 21


def test_draw_hand_skeletons_renders_each_detected_hand(monkeypatch):
    frame = object()
    hands = [[object() for _ in range(21)] for _ in range(2)]
    drawn_hands = []
    monkeypatch.setattr(
        spike.mp.tasks.vision.drawing_utils,
        "draw_landmarks",
        lambda _frame, hand, _connections: drawn_hands.append(hand),
    )

    draw_hand_skeletons(frame, hands)

    assert drawn_hands == hands
