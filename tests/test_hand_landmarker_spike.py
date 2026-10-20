from types import SimpleNamespace

from tools.diagnostics.hand_landmarker_spike import detection_rates, fingertip_pixels


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
