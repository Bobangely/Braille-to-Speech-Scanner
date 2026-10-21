"""Page format and 4-corner marker calibration for the mounted camera.

Coordinate system (page space):
  origin = top-left corner of the physical page, x to the right, y downward,
  units = millimetres. ``to_page`` / ``to_image`` convert between camera
  pixels and this space; ``warp_page`` renders it at a fixed px-per-mm.

Markers are black/white ArUco squares (DICT_4X4_50). They are unsaturated,
so the blue/red HSV dot mask in ``colored_braille`` ignores them, and the
marker ID fixes the corner (TL/TR/BR/BL) under any rotation.

A calibration is returned only when every check passes; otherwise
``PageCalibration.ok`` is False and no transform is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CORNERS = ("TL", "TR", "BR", "BL")


@dataclass(frozen=True)
class PageFormat:
    """Physical page layout. All lengths in millimetres."""

    width_mm: float = 210.0          # A5 landscape (half of a portrait A4)
    height_mm: float = 148.0
    margin_mm: float = 10.0          # page edge -> marker outer edge
    marker_size_mm: float = 15.0     # black border included
    marker_ids: tuple[int, int, int, int] = (0, 1, 2, 3)  # TL, TR, BR, BL
    aruco_dictionary: int = cv2.aruco.DICT_4X4_50
    # Colored Braille lines: SOV_braille_only 30 pt + 1.4 pt expanded in Word.
    dot_spacing_mm: float = 2.33     # dot 1 -> dot 4 and dot 1 -> dot 2
    cell_pitch_mm: float = 6.21      # cell to cell
    dot_diameter_mm: float = 1.5
    # Dot 1 of the first cell of each colored line; a raised line sits under each.
    line_origins_mm: tuple[tuple[float, float], ...] = ((30.0, 35.0), (30.0, 70.0), (30.0, 105.0))
    cells_per_line: int = 24
    raised_band_mm: float = 12.0     # height of the raised line under the last colored line

    @property
    def line_pairs(self) -> int:
        return len(self.line_origins_mm)

    def marker_corners_mm(self) -> dict[int, np.ndarray]:
        """Each marker's 4 corners (ArUco order TL,TR,BR,BL) in page mm."""
        s, m = self.marker_size_mm, self.margin_mm
        origins = (
            (m, m),
            (self.width_mm - m - s, m),
            (self.width_mm - m - s, self.height_mm - m - s),
            (m, self.height_mm - m - s),
        )
        square = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)
        return {mid: square + o for mid, o in zip(self.marker_ids, origins)}

    def cell_slots_mm(self, cell_index: int, line: int = 0) -> dict[int, tuple[float, float]]:
        """Six dot-slot centres of one cell in page mm (dots 1-3 left, 4-6 right)."""
        if not 0 <= cell_index < self.cells_per_line:
            raise IndexError(cell_index)
        x0 = self.line_origins_mm[line][0] + cell_index * self.cell_pitch_mm
        y0 = self.line_origins_mm[line][1]
        d = self.dot_spacing_mm
        return {
            dot: (x0 + (dot - 1) // 3 * d, y0 + (dot - 1) % 3 * d)
            for dot in range(1, 7)
        }

    def cell_box_mm(self, cell_index: int, line: int = 0) -> tuple[float, float, float, float]:
        """Cell bounding box (x1, y1, x2, y2) in mm, half a gap of padding."""
        slots = np.array(list(self.cell_slots_mm(cell_index, line).values()))
        pad_x = (self.cell_pitch_mm - self.dot_spacing_mm) / 2
        pad_y = self.dot_spacing_mm / 2
        (x1, y1), (x2, y2) = slots.min(axis=0), slots.max(axis=0)
        return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)

    def validate(self) -> None:
        inner_x1 = self.margin_mm + self.marker_size_mm
        inner_x2 = self.width_mm - inner_x1
        for line in range(self.line_pairs):
            first = self.cell_box_mm(0, line)
            last = self.cell_box_mm(self.cells_per_line - 1, line)
            if first[0] < inner_x1 or last[2] > inner_x2:
                raise ValueError("braille line overlaps marker columns")
        if len(set(self.marker_ids)) != 4:
            raise ValueError("marker ids must be unique")


DEFAULT_FORMAT = PageFormat()


@dataclass
class PageCalibration:
    ok: bool
    reason: str | None = None
    homography: np.ndarray | None = None   # image px -> page mm
    inverse: np.ndarray | None = None      # page mm -> image px
    reprojection_error_mm: float | None = None
    markers: dict[str, np.ndarray] = field(default_factory=dict)  # corner name -> 4x2 px

    def to_page(self, points) -> np.ndarray:
        return _apply(self._require(self.homography), points)

    def to_image(self, points) -> np.ndarray:
        return _apply(self._require(self.inverse), points)

    def _require(self, matrix):
        if not self.ok or matrix is None:
            raise RuntimeError(f"unreliable calibration: {self.reason}")
        return matrix


def _apply(matrix, points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def _detector(fmt: PageFormat):
    dictionary = cv2.aruco.getPredefinedDictionary(fmt.aruco_dictionary)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_markers(image, fmt: PageFormat = DEFAULT_FORMAT) -> dict[int, list[np.ndarray]]:
    """All detections of the format's marker ids (lists keep duplicates)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector(fmt).detectMarkers(gray)
    found: dict[int, list[np.ndarray]] = {}
    if ids is None:
        return found
    for quad, mid in zip(corners, ids.ravel()):
        if int(mid) in fmt.marker_ids:
            found.setdefault(int(mid), []).append(quad.reshape(4, 2).astype(np.float64))
    return found


def calibrate(
    image,
    fmt: PageFormat = DEFAULT_FORMAT,
    *,
    max_error_mm: float = 1.0,
    max_size_ratio: float = 4.0,
) -> PageCalibration:
    """Detect all four markers and fit image->page homography, or fail safely."""
    found = detect_markers(image, fmt)
    missing = [CORNERS[i] for i, mid in enumerate(fmt.marker_ids) if mid not in found]
    if missing:
        return PageCalibration(False, "missing_marker:" + ",".join(missing))
    duplicated = [CORNERS[i] for i, mid in enumerate(fmt.marker_ids) if len(found[mid]) > 1]
    if duplicated:
        return PageCalibration(False, "duplicate_marker:" + ",".join(duplicated))

    quads = {mid: found[mid][0] for mid in fmt.marker_ids}
    markers = {CORNERS[i]: quads[mid] for i, mid in enumerate(fmt.marker_ids)}

    # Marker apparent size must be comparable (rejects a stray tiny/huge print).
    sides = [np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1).mean() for q in quads.values()]
    if min(sides) <= 2 or max(sides) / min(sides) > max_size_ratio:
        return PageCalibration(False, "inconsistent_marker_size", markers=markers)

    # Marker centres in TL,TR,BR,BL order must form a convex, non-mirrored quad.
    centres = np.float32([quads[mid].mean(axis=0) for mid in fmt.marker_ids])
    if not cv2.isContourConvex(centres) or cv2.contourArea(centres) <= 0:
        return PageCalibration(False, "invalid_corner_order", markers=markers)
    edge1, edge2 = centres[1] - centres[0], centres[3] - centres[0]
    if edge1[0] * edge2[1] - edge1[1] * edge2[0] <= 0:  # image y-down: TL->TR x TL->BL > 0
        return PageCalibration(False, "mirrored_page", markers=markers)

    reference = fmt.marker_corners_mm()
    src = np.concatenate([quads[mid] for mid in fmt.marker_ids])
    dst = np.concatenate([reference[mid] for mid in fmt.marker_ids])
    homography, _ = cv2.findHomography(src, dst, 0)
    if homography is None or not np.isfinite(homography).all():
        return PageCalibration(False, "homography_failed", markers=markers)
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return PageCalibration(False, "singular_homography", markers=markers)

    error = float(np.linalg.norm(_apply(homography, src) - dst, axis=1).max())
    if error > max_error_mm:
        return PageCalibration(False, "high_reprojection_error", markers=markers,
                               reprojection_error_mm=error)
    return PageCalibration(True, None, homography, inverse, error, markers)


def warp_page(image, calibration: PageCalibration, fmt: PageFormat = DEFAULT_FORMAT,
              px_per_mm: float = 4.0):
    """Render the calibrated page into fixed page space (px = mm * px_per_mm)."""
    scale = np.diag([px_per_mm, px_per_mm, 1.0])
    size = (round(fmt.width_mm * px_per_mm), round(fmt.height_mm * px_per_mm))
    return cv2.warpPerspective(image, scale @ calibration._require(calibration.homography),
                               size, flags=cv2.INTER_LINEAR)


def render_page(fmt: PageFormat = DEFAULT_FORMAT, cells=None, px_per_mm: float = 4.0,
                dot_color=(255, 0, 0), lines=None):
    """Printable template: markers + colored Braille lines.

    ``lines`` holds one list of dot sets per colored line (e.g. ``[[{1}, {1, 2}]]``);
    ``cells`` is the single-line shorthand. Empty sets are spaces. Only active
    dots are drawn, plus a light-grey guide box per cell.
    """
    fmt.validate()
    w, h = round(fmt.width_mm * px_per_mm), round(fmt.height_mm * px_per_mm)
    page = np.full((h, w, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(fmt.aruco_dictionary)
    side = round(fmt.marker_size_mm * px_per_mm)
    for mid, quad in fmt.marker_corners_mm().items():
        x, y = np.round(quad[0] * px_per_mm).astype(int)
        page[y:y + side, x:x + side] = cv2.cvtColor(
            cv2.aruco.generateImageMarker(dictionary, mid, side), cv2.COLOR_GRAY2BGR)
    if lines is None:
        lines = [cells] if cells is not None else [SAMPLE_CELLS] * fmt.line_pairs
    radius = max(1, round(fmt.dot_diameter_mm / 2 * px_per_mm))
    for line, line_cells in enumerate(lines[: fmt.line_pairs]):
        for index, dots in enumerate(line_cells[: fmt.cells_per_line]):
            x1, y1, x2, y2 = (round(v * px_per_mm) for v in fmt.cell_box_mm(index, line))
            cv2.rectangle(page, (x1, y1), (x2, y2), (225, 225, 225), 1)
            for dot, (sx, sy) in fmt.cell_slots_mm(index, line).items():
                if dot in dots:
                    cv2.circle(page, (round(sx * px_per_mm), round(sy * px_per_mm)),
                               radius, dot_color, -1, cv2.LINE_AA)
    return page


class CalibrationTracker:
    """Keep the last good calibration while markers are covered; flag page changes.

    A camera on a stand does not move, so a hand over some markers keeps the
    previous calibration. The page counts as changed when a new calibration
    moves the markers by more than ``move_mm``, or when no marker at all has
    been seen for ``lost_s`` seconds.
    """

    def __init__(self, fmt: PageFormat = DEFAULT_FORMAT, move_mm: float = 3.0, lost_s: float = 2.0):
        self.fmt, self.move_mm, self.lost_s = fmt, move_mm, lost_s
        self.current: PageCalibration | None = None
        self._seen_s = 0.0
        self._centres_mm = np.array([q.mean(axis=0) for q in fmt.marker_corners_mm().values()])

    def update(self, image, now_s: float) -> tuple[PageCalibration | None, bool]:
        found = calibrate(image, self.fmt)
        if found.reason != "missing_marker:" + ",".join(CORNERS):
            self._seen_s = now_s
        if found.ok:
            if self.current is None:
                self.current = found
                return found, True
            centres_px = np.array([found.markers[c].mean(axis=0) for c in CORNERS])
            drift = np.linalg.norm(self.current.to_page(centres_px) - self._centres_mm, axis=1).max()
            if drift > self.move_mm:
                self.current = found
                return found, True
            return self.current, False
        if self.current is not None and now_s - self._seen_s > self.lost_s:
            self.current = None
            return None, True
        return self.current, False


# Generic demo pattern (not a specific Thai word); replace with real cells.
SAMPLE_CELLS = [{1}, {1, 2}, {1, 4}, {1, 4, 5}, {1, 5}, set(), {1, 2, 4}, {1, 2, 4, 5}]
