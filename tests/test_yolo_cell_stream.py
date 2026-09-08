import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

from yolo_cell_stream import plan_cells, crop_cell, read_cell, pair_markers, CellStream
from thai_decoder import tokenize_thai, decode_thai
from decoder import decode_cells
from yolo_detector import YOLOBrailleDetector


def page(lines, spacing=40, pitch=95, line_pitch=128, angle=0):
    image = np.full((80+int(len(lines)*line_pitch)+100, 900, 3), 255, np.uint8)
    dots, expected = [], []
    for line_id, patterns in enumerate(lines):
        for i, pattern in enumerate(patterns):
            expected.append(frozenset(map(int, pattern)))
            for d in map(int, pattern):
                x, y = 60+i*pitch+(d > 3)*spacing, 60+line_id*line_pitch+((d-1) % 3)*spacing
                cv2.circle(image, (round(x), round(y)), 8, (255, 0, 0), -1)
                dots.append(dict(center=(x, y), bbox=(x-8, y-8, x+8, y+8),
                                 area=256, confidence=.95))
    if angle:
        transform = cv2.getRotationMatrix2D((450, image.shape[0]/2), angle, 1)
        image = cv2.warpAffine(image, transform, image.shape[1::-1], borderValue=(255, 255, 255))
        for dot in dots:
            dot['center'] = tuple(transform @ np.array([*dot['center'], 1]))
    return image, dots, expected


def test_crop_predictor(crops):
    """Image-based test double: confirms that only the isolated crop reaches stage 2."""
    all_dots = []
    for crop in crops:
        mask = cv2.inRange(crop, (200, 0, 0), (255, 60, 60))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = []
        for contour in contours:
            moment = cv2.moments(contour)
            if moment['m00'] > 2:
                dots.append(dict(center=(moment['m10']/moment['m00'], moment['m01']/moment['m00']),
                                 confidence=.95, area=moment['m00']))
        all_dots.append(dots)
    return all_dots


class CellStreamTests(unittest.TestCase):
    def test_adjacent_lines_never_merge_or_share_source_dots(self):
        for gap in (1.0, 1.2, 1.8, 2.5):
            for angle in (-6, 0, 6):
                with self.subTest(gap=gap, angle=angle):
                    image, dots, expected = page([['6', '13456', '16'], ['6', '1456', '256']],
                                                 line_pitch=(2+gap)*40, angle=angle)
                    cells = list(CellStream(test_crop_predictor, batch_size=2).iter_cells(image, dots))
                    self.assertEqual([c['dots'] for c in cells], expected)
                    self.assertEqual([c['line_id'] for c in cells], [0, 0, 0, 1, 1, 1])
                    ids = [i for c in cells for i in c['source_dot_ids']]
                    self.assertEqual(sorted(ids), list(range(len(dots))))

    def test_wide_printed_cell_pitch_keeps_internal_prefix_on_right(self):
        for pitch in (95, 148):
            image, dots, expected = page([['1236', '2456', '6', '13456', '356']], pitch=pitch)
            cells = list(CellStream(test_crop_predictor).iter_cells(image, dots))
            self.assertEqual([c['dots'] for c in cells], expected)
            symbols = list(pair_markers(cells))
            self.assertEqual(symbols[2][0]['symbol'], 'ญ')

    def test_marker_does_not_consume_next_line(self):
        image, dots, _ = page([['13456', '6'], ['13456', '16']])
        cells = list(CellStream(test_crop_predictor).iter_cells(image, dots))
        symbols = list(pair_markers(cells))
        self.assertTrue(all(len(group) == 1 for group in symbols))

    def test_decoder_preserves_lines_and_wide_pitch_symbols(self):
        for pitch in (95, 148):
            image, dots, _ = page([['6', '13456', '16'], ['6', '1456', '256']],
                                  pitch=pitch, angle=6)
            cells = [cell for group in pair_markers(
                CellStream(test_crop_predictor).iter_cells(image, dots)) for cell in group]
            self.assertEqual(decode_thai(cells), 'ญา\nภ้')
            self.assertEqual(decode_cells(cells, 'thai'), 'ญา\nภ้')
            tokens = tokenize_thai(cells)
            self.assertTrue(tokens[3]['line_break_before'])
            self.assertTrue(tokens[1]['consumed'])
            # A missing/blank cell separates words and cannot be consumed by a marker.
            separated = [dict(c) for c in cells[:2]]
            separated[1]['reading_x'] = separated[0]['reading_x'] + 2*pitch
            self.assertFalse(tokenize_thai(separated)[1]['consumed'])

    def test_only_ambiguous_leading_three_can_be_repaired(self):
        image, dots, _ = page([['6', '13456']])
        cells = list(CellStream(test_crop_predictor).iter_cells(image, dots))
        broken = dict(cells[0], dots=frozenset({3}), column_ambiguous=True)
        result = list(pair_markers([broken, cells[1]]))
        self.assertEqual(result[0][0]['dots'], frozenset({6}))
        self.assertEqual(result[0][0]['marker_repair'], 'leading_3_to_6')
        genuine = dict(broken, column_ambiguous=False)
        self.assertEqual(list(pair_markers([genuine, cells[1]]))[0][0]['dots'], frozenset({3}))
        middle = dict(broken, line_cell_index=2)
        self.assertEqual(list(pair_markers([middle, cells[1]]))[0][0]['dots'], frozenset({3}))

    def test_english_indicators_reset_at_line_boundary(self):
        cells = [dict(dots=frozenset(map(int, pattern)), line_id=line, reading_x=x,
                      cell_pitch=148) for pattern, line, x in (
                          ('3456', 0, 0), ('1', 0, 148), ('6', 0, 296), ('1', 1, 0))]
        self.assertEqual(decode_cells(cells, 'english'), '1\na')

    def test_repaired_marker_grid_matches_the_observed_dot(self):
        image, dots, _ = page([['6', '13456']], pitch=148)
        original = list(CellStream(test_crop_predictor).iter_cells(image, dots))
        self.assertEqual(original[0]['dots'], frozenset({3}))
        cells = [cell for group in pair_markers(original) for cell in group]
        repaired = cells[0]
        self.assertEqual(decode_thai(cells), 'ญ')
        np.testing.assert_allclose(repaired['grid']['slots'][6], dots[0]['center'], atol=.5)
        self.assertAlmostEqual(repaired['reading_x'], 80)
        self.assertEqual(repaired['read_pattern'], frozenset({3}))
        self.assertEqual(repaired['crop_quad'], original[0]['crop_quad'])
        self.assertEqual(original[0]['dots'], frozenset({3}))

    def test_empty_crop_is_not_replaced_with_overview_dots(self):
        image, dots, _ = page([['6', '13456']])
        cells = list(CellStream(lambda crops: [[] for _ in crops]).iter_cells(image, dots))
        self.assertTrue(all(c['dots'] == frozenset() for c in cells))
        self.assertTrue(all(c['crop_status'] == 'empty' for c in cells))

    def test_duplicate_box_on_tiny_dot_does_not_shrink_grid_pitch(self):
        image, dots, expected = page([['1245', '26']])
        image = cv2.resize(image, None, fx=.2, fy=.2, interpolation=cv2.INTER_AREA)
        dots = [dict(dot, center=tuple(.2*np.asarray(dot['center'])), area=dot['area']*.04)
                for dot in dots]
        duplicate = dict(dots[-1], center=(dots[-1]['center'][0]+1.5, dots[-1]['center'][1]), area=5.2)
        planned = plan_cells(dots+[duplicate], image.shape)
        self.assertEqual([cell['dots'] for cell in planned], expected)
        self.assertAlmostEqual(planned[0]['dot_spacing'], 8, delta=.5)

    def test_bounded_batches_and_explicit_cell_budget(self):
        image, dots, _ = page([['13456']*7])
        batch_sizes = []
        def predict(crops):
            batch_sizes.append(len(crops))
            return test_crop_predictor(crops)
        list(CellStream(predict, batch_size=3).iter_cells(image, dots))
        self.assertEqual(batch_sizes, [3, 3, 1])
        with self.assertRaises(ValueError):
            list(CellStream(predict, max_cells=2).iter_cells(image, dots))

    def test_yolo_stream_keeps_crop_confirmation_and_never_calls_cv_detection(self):
        image, dots, _ = page([['6', '13456']])
        with patch.object(YOLOBrailleDetector, '_load_model'):
            detector = YOLOBrailleDetector(mode='yolo', tile_size=0)
        detector.model = Mock(side_effect=lambda source, **kwargs: [object() for _ in source]
                              if isinstance(source, list) else [object()])
        # Overview contains a full letter; the crop model rejects all dots.
        with patch.object(detector, '_extract_yolo_boxes', side_effect=[dots, [], []]):
            cells, debug = detector.detect(image)
        self.assertTrue(all(not cell['dots'] for cell in cells))
        calls = detector.model.call_args_list
        self.assertEqual(calls[0].kwargs['conf'], .3)
        self.assertEqual(calls[1].kwargs['conf'], .35)
        self.assertEqual(debug['method'], 'yolo_cell_stream')

    def test_explicit_yolo_reports_missing_weights(self):
        with patch.object(YOLOBrailleDetector, '_load_model'):
            detector = YOLOBrailleDetector(mode='yolo')
        with self.assertRaisesRegex(RuntimeError, 'weights'):
            detector.detect(np.zeros((100, 100, 3), np.uint8))


if __name__ == '__main__':
    unittest.main()
