import cv2
import numpy as np
import pytest

from page_calibration import (
    DEFAULT_FORMAT, PageFormat, calibrate, detect_markers, render_page, warp_page,
)

PX = 4.0  # template px per mm
FMT = DEFAULT_FORMAT


@pytest.fixture(scope="module")
def page():
    return render_page(FMT, px_per_mm=PX)


def camera_view(page, dst_corners, size=(1280, 960), background=90):
    """Project the template onto a synthetic camera frame."""
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src, np.float32(dst_corners))
    frame = cv2.warpPerspective(page, matrix, size, borderValue=(background,) * 3)
    return frame, matrix


def rotated_corners(angle_deg, centre=(640, 480), half=(420, 296)):
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    base = np.array([[-half[0], -half[1]], [half[0], -half[1]],
                     [half[0], half[1]], [-half[0], half[1]]])
    return base @ rot.T + centre


def assert_consistent(calib, matrix, tol_mm=0.6):
    """Known page points must round-trip through the calibration."""
    probes_mm = np.array([[0, 0], [FMT.width_mm, FMT.height_mm],
                          *FMT.cell_slots_mm(0).values(), *FMT.cell_slots_mm(7).values()])
    in_image = cv2.perspectiveTransform((probes_mm * PX).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    assert np.abs(calib.to_page(in_image) - probes_mm).max() < tol_mm
    assert np.abs(calib.to_image(probes_mm) - in_image).max() < tol_mm * 3


def test_format_geometry():
    FMT.validate()
    slots = FMT.cell_slots_mm(1)
    assert slots[1] == (FMT.line_origin_mm[0] + FMT.cell_pitch_mm, FMT.line_origin_mm[1])
    assert slots[4][0] - slots[1][0] == FMT.dot_spacing_mm
    assert slots[3][1] - slots[1][1] == 2 * FMT.dot_spacing_mm
    with pytest.raises(IndexError):
        FMT.cell_slots_mm(FMT.cells_per_line)
    with pytest.raises(ValueError):
        PageFormat(line_origin_mm=(10.0, 70.0)).validate()


def test_markers_are_not_colored_dots(page):
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
    saturated = cv2.inRange(hsv, (0, 60, 15), (179, 255, 255))
    for quad in FMT.marker_corners_mm().values():
        x1, y1 = (quad[0] * PX).astype(int)
        x2, y2 = (quad[2] * PX).astype(int)
        assert saturated[y1:y2, x1:x2].sum() == 0


def test_frontal_page(page):
    frame, matrix = camera_view(page, rotated_corners(0))
    calib = calibrate(frame, FMT)
    assert calib.ok, calib.reason
    assert set(calib.markers) == {"TL", "TR", "BR", "BL"}
    assert_consistent(calib, matrix)


@pytest.mark.parametrize("angle", [15, -30, 90, 180, 250])
def test_rotated_page(page, angle):
    frame, matrix = camera_view(page, rotated_corners(angle, half=(300, 212)))
    calib = calibrate(frame, FMT)
    assert calib.ok, calib.reason
    assert_consistent(calib, matrix)
    # TL marker stays the physical TL corner whatever the rotation.
    tl_mm = calib.to_page(calib.markers["TL"].mean(axis=0))[0]
    assert tl_mm[0] < FMT.width_mm / 2 and tl_mm[1] < FMT.height_mm / 2


def test_tilted_perspective_page(page):
    corners = [[330, 180], [960, 220], [1180, 840], [140, 780]]
    frame, matrix = camera_view(page, corners)
    calib = calibrate(frame, FMT)
    assert calib.ok, calib.reason
    assert_consistent(calib, matrix, tol_mm=0.8)


def test_dim_noisy_lighting(page):
    frame, matrix = camera_view(page, [[300, 200], [990, 230], [1060, 760], [230, 720]])
    rng = np.random.default_rng(0)
    gradient = np.linspace(0.35, 0.9, frame.shape[1])[None, :, None]
    dim = np.clip(frame * gradient + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
    calib = calibrate(dim, FMT)
    assert calib.ok, calib.reason
    assert_consistent(calib, matrix, tol_mm=0.8)


def test_warp_normalizes_cells(page):
    frame, _ = camera_view(page, [[330, 180], [960, 220], [1180, 840], [140, 780]])
    calib = calibrate(frame, FMT)
    flat = warp_page(frame, calib, FMT, px_per_mm=PX)
    assert flat.shape[:2] == page.shape[:2]
    # Dot 1 of cell 0 must be blue at its page coordinate after normalization.
    x, y = (np.array(FMT.cell_slots_mm(0)[1]) * PX).round().astype(int)
    b, g, r = flat[y, x].astype(int)
    assert b > 150 and r < 100 and g < 100


@pytest.mark.parametrize("corner", ["TL", "BR"])
def test_missing_marker_is_rejected(page, corner):
    covered = page.copy()
    index = ("TL", "TR", "BR", "BL").index(corner)
    quad = FMT.marker_corners_mm()[FMT.marker_ids[index]] * PX
    (x1, y1), (x2, y2) = quad[0].astype(int), quad[2].astype(int)
    covered[y1 - 4:y2 + 4, x1 - 4:x2 + 4] = 255  # e.g. hand over marker
    frame, _ = camera_view(covered, rotated_corners(10))
    calib = calibrate(frame, FMT)
    assert not calib.ok and calib.reason == f"missing_marker:{corner}"
    with pytest.raises(RuntimeError):
        calib.to_page([[0, 0]])
    with pytest.raises(RuntimeError):
        warp_page(frame, calib, FMT)


def test_partially_out_of_frame_is_rejected(page):
    frame, _ = camera_view(page, [[-250, 150], [800, 150], [800, 800], [-250, 800]])
    calib = calibrate(frame, FMT)
    assert not calib.ok and calib.reason.startswith("missing_marker")


def test_duplicate_marker_is_rejected(page):
    dup = page.copy()
    tl = FMT.marker_corners_mm()[FMT.marker_ids[0]] * PX
    (x1, y1), (x2, y2) = tl[0].astype(int), tl[2].astype(int)
    dup[300:300 + (y2 - y1), 400:400 + (x2 - x1)] = page[y1:y2, x1:x2]
    frame, _ = camera_view(dup, rotated_corners(0))
    calib = calibrate(frame, FMT)
    assert not calib.ok and calib.reason == "duplicate_marker:TL"


def test_swapped_markers_are_rejected(page):
    swapped = page.copy()
    q = FMT.marker_corners_mm()
    (ax, ay), (bx, by) = (q[0][0] * PX).astype(int), (q[1][0] * PX).astype(int)
    s = round(FMT.marker_size_mm * PX)
    a, b = page[ay:ay + s, ax:ax + s].copy(), page[by:by + s, bx:bx + s].copy()
    swapped[ay:ay + s, ax:ax + s], swapped[by:by + s, bx:bx + s] = b, a
    frame, _ = camera_view(swapped, rotated_corners(0))
    calib = calibrate(frame, FMT)
    assert not calib.ok and calib.reason in {"invalid_corner_order", "mirrored_page"}


def test_mirrored_view_is_rejected(page):
    frame, _ = camera_view(cv2.flip(page, 1), rotated_corners(0))
    calib = calibrate(frame, FMT)
    assert not calib.ok


def test_stray_small_marker_size_is_rejected(page):
    frame, _ = camera_view(page, rotated_corners(0))
    small = cv2.resize(page, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    s = small.shape[:2]
    frame2 = frame.copy()
    # Replace the real BL marker with a tiny one: geometry/size check must fail.
    bl = calibrate(frame, FMT).markers["BL"].astype(int)
    (x1, y1), (x2, y2) = bl.min(axis=0) - 6, bl.max(axis=0) + 6
    frame2[y1:y2, x1:x2] = 255
    m = FMT.marker_corners_mm()[FMT.marker_ids[3]] * PX * 0.15
    (mx1, my1), (mx2, my2) = m[0].astype(int) - 2, m[2].astype(int) + 2
    patch = small[my1:my2, mx1:mx2]
    frame2[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]] = patch
    calib = calibrate(frame2, FMT)
    assert not calib.ok, s


def test_no_markers_and_grayscale_input():
    blank = np.full((480, 640), 200, np.uint8)
    assert detect_markers(blank, FMT) == {}
    calib = calibrate(blank, FMT)
    assert not calib.ok and calib.reason == "missing_marker:TL,TR,BR,BL"
