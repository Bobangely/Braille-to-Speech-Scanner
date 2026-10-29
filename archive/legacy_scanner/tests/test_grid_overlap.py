"""Ownership and neighbour geometry invariants before Braille recognition."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cell_geometry import slot_points, validate_cell_grids
from cell_tracking import CellGridTracker, transform_grid
from colored_braille import ColoredCellStream
from tests.test_yolo_cell_stream import page
from yolo_cell_stream import plan_cells, read_cell


class GridOverlapTests(unittest.TestCase):
    @staticmethod
    def uneven_page(scale=1., lines=1, color=(255, 0, 0)):
        # Vertical pitch is smaller than horizontal pitch. A sparse painted
        # cell has a slightly wider column gap than the complete neighbours.
        patterns = ['123456', '145', '123456', '3', '123456']
        image = np.full((round((140+lines*180)*scale), round(800*scale), 3), 255, np.uint8)
        for line in range(lines):
            for cell, pattern in enumerate(patterns):
                for dot in map(int, pattern):
                    width = 51 if cell == 1 else 46
                    x = (60+125*cell+(dot > 3)*width)*scale
                    y = (60+180*line+(dot-1)%3*40)*scale
                    cv2.circle(image, (round(x), round(y)), max(2, round(6*scale)), color, -1)
        return image, [frozenset(map(int, p)) for p in patterns]*lines

    def assert_unique_valid(self, cells):
        sources = [source for cell in cells for source in cell['source_dot_ids']]
        self.assertEqual(len(sources), len(set(sources)))
        for cell in cells:
            self.assertEqual(cell['geometry_status'], 'valid')
            self.assertEqual(slot_points(cell).shape, (6, 2))
            self.assertEqual(cell['overlap_ratio'], 0.)
            self.assertEqual(cell['geometry_debug']['source_dots'], cell['source_dot_ids'])

    def test_horizontal_scale_is_fitted_per_line_before_sparse_grouping(self):
        for scale in (.35, .5, 1., 1.5):
            for lines in (1, 2, 4):
                for color, bgr in [('blue', (255, 0, 0)), ('red', (0, 0, 255))]:
                    with self.subTest(scale=scale, lines=lines, color=color):
                        image, expected = self.uneven_page(scale, lines, bgr)
                        cells, debug = ColoredCellStream(color).detect(image)
                        self.assertEqual([c['dots'] for c in cells], expected)
                        self.assertEqual(debug['line_count'], lines)
                        self.assert_unique_valid(cells)

    def test_single_image_repeats_deterministically(self):
        image, expected = self.uneven_page(lines=4)
        reader = ColoredCellStream()
        first, _ = reader.detect(image)
        for _ in range(8):
            cells, _ = reader.detect(image)
            np.testing.assert_array_equal([slot_points(c) for c in cells], [slot_points(c) for c in first])
            self.assertEqual([c['source_dot_ids'] for c in cells], [c['source_dot_ids'] for c in first])
        for _ in range(8):
            cells, debug = reader.detect(image, context='camera')
            self.assertEqual([c['dots'] for c in cells], expected)
            self.assertFalse(debug['grid_pending'])
            np.testing.assert_allclose([slot_points(c) for c in cells], [slot_points(c) for c in first], atol=.01)
            self.assert_unique_valid(cells)

    def test_overlap_is_invalid_without_guessing_which_cell_is_correct(self):
        image, dots, _ = page([['123456', '123456']])
        cells = plan_cells(dots, image.shape)
        cells[1] = transform_grid(cells[1], np.array([[1., 0, -75], [0, 1., 0], [0, 0, 1.]]), image.shape)
        result = validate_cell_grids(cells, image.shape)
        for cell in result:
            self.assertEqual(cell['geometry_error'], 'overlapping_grid')
            self.assertGreater(cell['overlap_ratio'], 0.)
            self.assertFalse(read_cell(cell, [], np.eye(3))['dots'])

    def test_source_dot_cannot_be_owned_by_two_cells_even_on_other_lines(self):
        image, dots, _ = page([['123456'], ['123456']])
        cells = plan_cells(dots, image.shape)
        cells[1]['source_dot_ids'].append(cells[0]['source_dot_ids'][0])
        result = validate_cell_grids(cells, image.shape)
        self.assertEqual([c['geometry_error'] for c in result], ['shared_source_dot']*2)

    def test_repair_cannot_cover_inactive_slots_of_a_neighbour(self):
        image, dots, _ = page([['123456', '1', '123456']])
        cells = plan_cells(dots, image.shape)
        # Move sparse cell to the neighbour's empty column and lose its slots.
        neighbour = deepcopy(cells[0])
        neighbour['dots'] = frozenset({1})
        damaged = deepcopy(cells[1])
        damaged['center'] = tuple(np.asarray(neighbour['center'])+(20., 0.))
        damaged['grid']['slots'] = {1: damaged['center']}
        fixed = validate_cell_grids([neighbour, damaged, cells[2]], image.shape)[1]
        self.assertEqual(fixed['geometry_status'], 'invalid')
        self.assertNotEqual(fixed['geometry_status'], 'repaired')

    def test_duplicate_candidate_centers_still_have_single_cell_ownership(self):
        image, dots, _ = page([['123456', '145', '123456']])
        dots.extend(deepcopy(dots[:3]))
        cells = plan_cells(dots, image.shape)
        self.assert_unique_valid(cells)
        self.assertEqual(len(cells), 3)

    def test_tracker_revalidates_geometry_after_motion(self):
        image, dots, _ = page([['123456', '123456']])
        reader = ColoredCellStream()
        reader.detect(image, context='camera')
        measured = plan_cells(dots, image.shape)
        predicted = deepcopy(measured)
        # A bad motion/layout proposal preserves six slots, but stretches the
        # first cell into the second. Keep current evidence in both cells.
        for dot in (4, 5, 6):
            x, y = predicted[0]['grid']['slots'][dot]
            predicted[0]['grid']['slots'][dot] = (x+60, y)
        predicted[0]['crop_quad'] = [[42, 42], [178, 42], [178, 158], [42, 158]]
        with patch('cell_tracking.estimate_motion', return_value=np.eye(3)), \
                patch('cell_tracking.transform_grid', side_effect=predicted), \
                patch.object(CellGridTracker, '_associate', return_value={}):
            cells, debug = reader.detect(image, context='camera')
        self.assertEqual(debug['grid_tracking'], 'invalid')
        self.assertTrue(any(c['geometry_error'] == 'overlapping_grid' for c in cells))
        self.assertFalse(reader._grid_tracker.cells)


if __name__ == '__main__':
    unittest.main()
