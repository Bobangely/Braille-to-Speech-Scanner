import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from page_calibration import (
    DEFAULT_FORMAT, CalibrationTracker, PageFormat, calibrate, detect_markers, render_page, warp_page,
)

PX = 4.0  # template px per mm
FMT = DEFAULT_FORMAT


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
                          *FMT.cell_slots_mm(0).values(), *FMT.cell_slots_mm(7, line=2).values()])
    in_image = cv2.perspectiveTransform((probes_mm * PX).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    assert np.abs(calib.to_page(in_image) - probes_mm).max() < tol_mm
    assert np.abs(calib.to_image(probes_mm) - in_image).max() < tol_mm * 3


def cover_marker(page, corner, pad=4):
    """White-out one marker, e.g. a hand over it."""
    covered = page.copy()
    index = ("TL", "TR", "BR", "BL").index(corner)
    quad = FMT.marker_corners_mm()[FMT.marker_ids[index]] * PX
    (x1, y1), (x2, y2) = quad[0].astype(int), quad[2].astype(int)
    covered[y1 - pad:y2 + pad, x1 - pad:x2 + pad] = 255
    return covered


class PageCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = render_page(FMT, px_per_mm=PX)

    def test_default_format_is_a5_landscape(self):
        self.assertEqual((FMT.width_mm, FMT.height_mm), (210.0, 148.0))
        self.assertEqual((FMT.marker_size_mm, FMT.margin_mm), (15.0, 10.0))
        self.assertEqual(FMT.line_pairs, 3)
        FMT.validate()

    def test_format_geometry(self):
        slots = FMT.cell_slots_mm(1)
        origin = FMT.line_origins_mm[0]
        self.assertEqual(slots[1], (origin[0] + FMT.cell_pitch_mm, origin[1]))
        self.assertAlmostEqual(slots[4][0] - slots[1][0], FMT.dot_spacing_mm)
        self.assertAlmostEqual(slots[3][1] - slots[1][1], 2 * FMT.dot_spacing_mm)
        self.assertEqual(FMT.cell_slots_mm(0, line=1)[1], FMT.line_origins_mm[1])
        with self.assertRaises(IndexError):
            FMT.cell_slots_mm(FMT.cells_per_line)
        with self.assertRaises(ValueError):
            PageFormat(line_origins_mm=((10.0, 70.0),)).validate()

    def test_markers_are_not_colored_dots(self):
        hsv = cv2.cvtColor(self.page, cv2.COLOR_BGR2HSV)
        saturated = cv2.inRange(hsv, (0, 60, 15), (179, 255, 255))
        for quad in FMT.marker_corners_mm().values():
            x1, y1 = (quad[0] * PX).astype(int)
            x2, y2 = (quad[2] * PX).astype(int)
            self.assertEqual(saturated[y1:y2, x1:x2].sum(), 0)

    def test_frontal_page(self):
        frame, matrix = camera_view(self.page, rotated_corners(0))
        calib = calibrate(frame, FMT)
        self.assertTrue(calib.ok, calib.reason)
        self.assertEqual(set(calib.markers), {"TL", "TR", "BR", "BL"})
        assert_consistent(calib, matrix)

    def test_rotated_page(self):
        for angle in (15, -30, 90, 180, 250):
            with self.subTest(angle=angle):
                frame, matrix = camera_view(self.page, rotated_corners(angle, half=(300, 212)))
                calib = calibrate(frame, FMT)
                self.assertTrue(calib.ok, calib.reason)
                assert_consistent(calib, matrix)
                # TL marker stays the physical TL corner whatever the rotation.
                tl_mm = calib.to_page(calib.markers["TL"].mean(axis=0))[0]
                self.assertTrue(tl_mm[0] < FMT.width_mm / 2 and tl_mm[1] < FMT.height_mm / 2)

    def test_tilted_perspective_page(self):
        frame, matrix = camera_view(self.page, [[330, 180], [960, 220], [1180, 840], [140, 780]])
        calib = calibrate(frame, FMT)
        self.assertTrue(calib.ok, calib.reason)
        assert_consistent(calib, matrix, tol_mm=0.8)

    def test_dim_noisy_lighting(self):
        frame, matrix = camera_view(self.page, [[300, 200], [990, 230], [1060, 760], [230, 720]])
        rng = np.random.default_rng(0)
        gradient = np.linspace(0.35, 0.9, frame.shape[1])[None, :, None]
        dim = np.clip(frame * gradient + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
        calib = calibrate(dim, FMT)
        self.assertTrue(calib.ok, calib.reason)
        assert_consistent(calib, matrix, tol_mm=0.8)

    def test_warp_normalizes_cells(self):
        frame, _ = camera_view(self.page, [[330, 180], [960, 220], [1180, 840], [140, 780]])
        calib = calibrate(frame, FMT)
        flat = warp_page(frame, calib, FMT, px_per_mm=PX)
        self.assertEqual(flat.shape[:2], self.page.shape[:2])
        # Dot 1 of cell 0 must be blue at its page coordinate after normalization.
        x, y = (np.array(FMT.cell_slots_mm(0)[1]) * PX).round().astype(int)
        b, g, r = flat[y, x].astype(int)
        self.assertTrue(b > 150 and r < 100 and g < 100)

    def test_missing_marker_is_rejected(self):
        for corner in ("TL", "BR"):
            with self.subTest(corner=corner):
                frame, _ = camera_view(cover_marker(self.page, corner), rotated_corners(10))
                calib = calibrate(frame, FMT)
                self.assertFalse(calib.ok)
                self.assertEqual(calib.reason, f"missing_marker:{corner}")
                with self.assertRaises(RuntimeError):
                    calib.to_page([[0, 0]])
                with self.assertRaises(RuntimeError):
                    warp_page(frame, calib, FMT)

    def test_partially_out_of_frame_is_rejected(self):
        frame, _ = camera_view(self.page, [[-250, 150], [800, 150], [800, 800], [-250, 800]])
        calib = calibrate(frame, FMT)
        self.assertFalse(calib.ok)
        self.assertTrue(calib.reason.startswith("missing_marker"))

    def test_duplicate_marker_is_rejected(self):
        dup = self.page.copy()
        tl = FMT.marker_corners_mm()[FMT.marker_ids[0]] * PX
        (x1, y1), (x2, y2) = tl[0].astype(int), tl[2].astype(int)
        dup[300:300 + (y2 - y1), 400:400 + (x2 - x1)] = self.page[y1:y2, x1:x2]
        frame, _ = camera_view(dup, rotated_corners(0))
        calib = calibrate(frame, FMT)
        self.assertFalse(calib.ok)
        self.assertEqual(calib.reason, "duplicate_marker:TL")

    def test_swapped_markers_are_rejected(self):
        swapped = self.page.copy()
        q = FMT.marker_corners_mm()
        (ax, ay), (bx, by) = (q[0][0] * PX).astype(int), (q[1][0] * PX).astype(int)
        s = round(FMT.marker_size_mm * PX)
        a, b = self.page[ay:ay + s, ax:ax + s].copy(), self.page[by:by + s, bx:bx + s].copy()
        swapped[ay:ay + s, ax:ax + s], swapped[by:by + s, bx:bx + s] = b, a
        frame, _ = camera_view(swapped, rotated_corners(0))
        calib = calibrate(frame, FMT)
        self.assertFalse(calib.ok)
        self.assertIn(calib.reason, {"invalid_corner_order", "mirrored_page"})

    def test_mirrored_view_is_rejected(self):
        frame, _ = camera_view(cv2.flip(self.page, 1), rotated_corners(0))
        self.assertFalse(calibrate(frame, FMT).ok)

    def test_stray_small_marker_size_is_rejected(self):
        frame, _ = camera_view(self.page, rotated_corners(0))
        small = cv2.resize(self.page, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
        frame2 = frame.copy()
        # Replace the real BL marker with a tiny one: geometry/size check must fail.
        bl = calibrate(frame, FMT).markers["BL"].astype(int)
        (x1, y1), (x2, y2) = bl.min(axis=0) - 6, bl.max(axis=0) + 6
        frame2[y1:y2, x1:x2] = 255
        m = FMT.marker_corners_mm()[FMT.marker_ids[3]] * PX * 0.15
        (mx1, my1), (mx2, my2) = m[0].astype(int) - 2, m[2].astype(int) + 2
        patch = small[my1:my2, mx1:mx2]
        frame2[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]] = patch
        self.assertFalse(calibrate(frame2, FMT).ok)

    def test_no_markers_and_grayscale_input(self):
        blank = np.full((480, 640), 200, np.uint8)
        self.assertEqual(detect_markers(blank, FMT), {})
        calib = calibrate(blank, FMT)
        self.assertFalse(calib.ok)
        self.assertEqual(calib.reason, "missing_marker:TL,TR,BR,BL")

    def test_export_markers_writes_four_files(self):
        from tools.export_markers import export_markers
        with tempfile.TemporaryDirectory() as out:
            paths = export_markers(Path(out), px=200, fmt=FMT)
            self.assertEqual(sorted(p.name for p in paths),
                             ["marker_0_TL.png", "marker_1_TR.png", "marker_2_BR.png", "marker_3_BL.png"])
            for path in paths:
                image = cv2.copyMakeBorder(cv2.imread(str(path)), 40, 40, 40, 40,
                                           cv2.BORDER_CONSTANT, value=(255, 255, 255))
                self.assertEqual(list(detect_markers(image, FMT)), [int(path.name.split("_")[1])])


class CalibrationTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = render_page(FMT, px_per_mm=PX)
        cls.frame, _ = camera_view(cls.page, rotated_corners(0))
        cls.px_per_mm = 840 / FMT.width_mm  # rotated_corners(0) spans 840 px across the page

    def shifted(self, mm):
        frame, _ = camera_view(self.page, rotated_corners(0, centre=(640 + mm * self.px_per_mm, 480)))
        return frame

    def test_first_calibration_is_a_new_page(self):
        calib, changed = CalibrationTracker(FMT).update(self.frame, 0.0)
        self.assertTrue(calib.ok and changed)

    def test_partial_occlusion_keeps_last_calibration(self):
        tracker = CalibrationTracker(FMT)
        first, _ = tracker.update(self.frame, 0.0)
        covered, _ = camera_view(cover_marker(self.page, "TR"), rotated_corners(0))
        for now in (0.5, 3.0, 10.0):
            calib, changed = tracker.update(covered, now)
            self.assertIs(calib, first)
            self.assertFalse(changed)

    def test_marker_shift_over_3mm_flags_page_change(self):
        tracker = CalibrationTracker(FMT)
        tracker.update(self.frame, 0.0)
        _, changed = tracker.update(self.shifted(1), 0.1)
        self.assertFalse(changed)
        _, changed = tracker.update(self.shifted(4), 0.2)
        self.assertTrue(changed)

    def test_markers_lost_over_2s_flags_page_change(self):
        blank = np.full_like(self.frame, 90)
        tracker = CalibrationTracker(FMT)
        tracker.update(self.frame, 0.0)
        calib, changed = tracker.update(blank, 1.9)
        self.assertTrue(calib is not None and not changed)
        calib, changed = tracker.update(blank, 2.1)
        self.assertTrue(calib is None and changed)


if __name__ == "__main__":
    unittest.main()
