"""Sparse cells retain a complete logical grid before dot assignment."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

from cell_geometry import grid_error, slot_points, validate_cell_grids
from cell_tracking import transform_grid
from colored_braille import ColoredCellStream
from decoder import decode_cells, decode_cells_verbose
from live_preview import scaled_geometry
from tests.test_yolo_cell_stream import page
from yolo_cell_stream import CellStream, plan_cells, read_cell


class CellGeometryTests(unittest.TestCase):
    def assert_complete(self, cell, shape):
        self.assertIsNone(grid_error(cell, shape))
        points = slot_points(cell)
        self.assertEqual(len(np.unique(points, axis=0)), 6)
        x1, y1, x2, y2 = cell['grid']['bbox']
        self.assertTrue(np.all(points >= (x1, y1)))
        self.assertTrue(np.all(points <= (x2, y2)))

    def test_one_to_three_active_dots_keep_six_slots_across_lines_and_scales(self):
        patterns = ['123456', '1', '4', '23', '56', '135', '246', '123456']
        for lines in (1, 2, 3, 4):
            for scale in (.2, .35, .5, 1., 1.5):
                for angle in (-4, 0, 4):
                    with self.subTest(lines=lines, scale=scale, angle=angle):
                        image, _, expected = page([patterns]*lines, angle=angle, line_pitch=180)
                        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                        cells, _ = ColoredCellStream().detect(image)
                        self.assertEqual([c['dots'] for c in cells], expected)
                        self.assertEqual([c['line_id'] for c in cells], [i for i in range(lines) for _ in patterns])
                        for cell in cells:
                            self.assert_complete(cell, image.shape)

    def test_sparse_column_is_not_paired_with_the_next_dense_cell(self):
        patterns = ['123456', '1', '123456', '4', '123456', '26', '123456', '3', '123456']
        left = np.r_[60, 60+np.cumsum(np.random.default_rng(0).uniform(86, 126, 8))]
        for scale in (.2, .5, 1., 1.5):
            with self.subTest(scale=scale):
                dots = []
                owners = []
                for cell_id, (x, pattern) in enumerate(zip(left, patterns)):
                    for d in map(int, pattern):
                        centre = (scale*(x+(d > 3)*40), scale*(60+((d-1) % 3)*40))
                        dots.append(dict(center=centre, area=64*scale**2, confidence=.95))
                        owners.append(cell_id)
                shape = (round(260*scale), round(1200*scale), 3)
                cells = plan_cells(dots, shape)
                self.assertEqual(len(cells), len(patterns))
                for cell_id, (cell, pattern) in enumerate(zip(cells, patterns)):
                    self.assert_complete(cell, shape)
                    self.assertEqual({owners[i] for i in cell['source_dot_ids']}, {cell_id})
                    read = read_cell(cell, [dots[i] for i in cell['source_dot_ids']], np.eye(3))
                    self.assertEqual(read['dots'], frozenset(map(int, pattern)))

    def test_lost_slots_and_collapsed_crop_recover_from_same_line_neighbours(self):
        image, dots, _ = page([['123456', '1', '123456']])
        original = plan_cells(dots, image.shape)
        for damage in ('missing', 'collapsed', 'crop_only'):
            with self.subTest(damage=damage):
                cells = deepcopy(original)
                target = cells[1]
                point = target['grid']['slots'][1]
                if damage != 'crop_only':
                    target['grid']['slots'] = ({1: point} if damage == 'missing'
                                              else {i: point for i in range(1, 7)})
                x, y = point
                target['crop_quad'] = [(x-4, y-4), (x+4, y-4), (x+4, y+4), (x-4, y+4)]
                source = deepcopy(cells)
                fixed = validate_cell_grids(cells, image.shape)[1]
                self.assertEqual(fixed['geometry_status'], 'repaired')
                self.assert_complete(fixed, image.shape)
                read = read_cell(fixed, [dots[i] for i in fixed['source_dot_ids']], np.eye(3))
                self.assertEqual(read['dots'], frozenset({1}))
                self.assertEqual(cells, source)

    def test_unrepairable_grid_is_unknown_and_does_not_read_color_evidence(self):
        image, dots, _ = page([['123456', '1', '123456']])
        cells = plan_cells(dots, image.shape)
        broken = deepcopy(cells[1])
        broken['grid']['slots'] = {1: broken['grid']['slots'][1]}
        other_line = deepcopy(cells[0])
        other_line['line_id'] = 1
        fixed = validate_cell_grids([broken, other_line], image.shape)[0]
        self.assertEqual(fixed['geometry_status'], 'invalid')
        read = read_cell(fixed, [dict(center='must not be sampled')], np.eye(3))
        self.assertFalse(read['dots'])
        self.assertEqual(read['crop_status'], 'invalid_grid')
        self.assertEqual(decode_cells([read], 'thai'), '�')
        self.assertEqual(decode_cells([read], 'english'), '�')
        self.assertEqual(decode_cells_verbose([read], 'thai')[0]['warning'], 'invalid_cell_grid')
        predictor = Mock(side_effect=AssertionError('invalid crop reached YOLO'))
        with patch('yolo_cell_stream.plan_cells', return_value=[fixed]):
            result = list(CellStream(predictor).iter_cells(image, dots))
        self.assertEqual(result[0]['crop_status'], 'invalid_grid')
        predictor.assert_not_called()

    def test_slot_ids_are_used_even_when_dictionary_order_changes(self):
        image, dots, expected = page([['123456', '4', '123456']])
        cells = plan_cells(dots, image.shape)
        target = deepcopy(cells[1])
        target['grid']['slots'] = dict(reversed(list(target['grid']['slots'].items())))
        result = read_cell(target, [dots[i] for i in target['source_dot_ids']], np.eye(3))
        self.assertEqual(result['dots'], expected[1])

    def test_repair_does_not_cross_lines_or_guess_ambiguous_phase(self):
        image, dots, _ = page([['123456', '1', '123456']])
        original = plan_cells(dots, image.shape)
        for reason in ('row_ambiguous', 'column_ambiguous', 'bad_center', 'other_line'):
            with self.subTest(reason=reason):
                cells = deepcopy(original)
                target = cells[1]
                target['grid']['slots'] = {1: target['grid']['slots'][1]}
                if reason == 'bad_center':
                    target['center'] = 'invalid'
                elif reason == 'other_line':
                    shift = original[1]['dot_spacing']/2
                    neighbour = transform_grid(original[1],
                        np.array([[1., 0, 0], [0, 1., shift], [0, 0, 1.]]), image.shape)
                    neighbour['line_id'] = 1
                    cells.append(neighbour)
                else:
                    target[reason] = True
                fixed = validate_cell_grids(cells, image.shape)[1]
                self.assertEqual(fixed['geometry_status'], 'invalid')
                self.assertFalse(read_cell(fixed, [], np.eye(3))['dots'])

    def test_motion_and_preview_scaling_keep_full_logical_bounds(self):
        image, dots, _ = page([['123456', '1', '123456']])
        cell = plan_cells(dots, image.shape)[1]
        x, y = cell['grid']['slots'][1]
        cell['grid']['bbox'] = (x-4, y-4, x+4, y+4)
        transformed = transform_grid(cell, np.array([[1., 0, 5], [0, 1., 3], [0, 0, 1.]]), image.shape)
        self.assert_complete(transformed, image.shape)
        _, small = scaled_geometry([], [transformed], .25, .25)
        self.assert_complete(small[0], (round(image.shape[0]*.25), round(image.shape[1]*.25), 3))


if __name__ == '__main__':
    unittest.main()
