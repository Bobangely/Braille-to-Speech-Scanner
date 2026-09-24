"""Live HUD keeps all measured patterns visible without changing recognition."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import ImageDraw

from decoder import decode_cells, decode_cells_verbose
from live_preview import LivePreview, scaled_geometry
from tests.test_yolo_cell_stream import page
from yolo_detector import YOLOBrailleDetector


class CellHUDTests(unittest.TestCase):
    def setUp(self):
        self.reader = YOLOBrailleDetector()
        image, _, _ = page([['135', '123456', '6', '13456'], ['1245', '16']])
        self.image = image
        self.cells, self.debug = self.reader.detect(image, lang='thai')

    def draw(self, cells=None, details=False):
        return self.reader.annotate_with_text(self.image, self.debug['dots'],
            self.cells if cells is None else cells, details=details, footer=False, cell_hud=True)

    def test_default_live_preview_draws_labels_and_patterns_without_number_clutter(self):
        preview = LivePreview(dashboard=True)
        result = dict(result_id=1, cells=self.cells, dots=self.debug['dots'],
                      decoded_text='', verbose_results=[])
        with patch('yolo_detector.cv2.putText', wraps=cv2.putText) as text, \
             patch.object(ImageDraw.ImageDraw, 'text', autospec=True,
                          side_effect=ImageDraw.ImageDraw.text) as unicode_text:
            preview.render(self.image, self.reader, result, 'thai')
        labels = [call.args[1] for call in text.call_args_list if call.args[6] == 1]
        headers = [call.args[2] for call in unicode_text.call_args_list]
        for i in range(1, len(self.cells)+1):
            self.assertIn(f'C{i}', headers)
        self.assertEqual(headers.count('…'), len(self.cells))
        for dot in range(1, 7):
            self.assertNotIn(str(dot), labels)
        self.assertIn('[1,3,5]', labels)
        # All six bits must remain available even if the pattern wraps.
        pattern = ''.join(value for value in labels if '[' in value or ']' in value)
        self.assertIn('[1,2,3,4,5,6]', pattern)

    def test_markers_use_existing_rotated_slots_and_distinguish_active_inactive(self):
        cell = deepcopy(self.cells[0])
        cell['track_id'] = 81
        # A display geometry with skew cannot be reconstructed from bbox thirds.
        cell['grid']['slots'] = {dot: (x+(dot % 3)*3, y+(dot > 3)*5)
                                for dot, (x, y) in cell['grid']['slots'].items()}
        with patch('yolo_detector.cv2.circle', wraps=cv2.circle) as circle, \
             patch.object(ImageDraw.ImageDraw, 'text', autospec=True,
                          side_effect=ImageDraw.ImageDraw.text) as text:
            self.draw([cell])
        self.assertTrue(all(call.args[4] > 0 for call in circle.call_args_list))
        rings = {call.args[1]: (call.args[2], call.args[3]) for call in circle.call_args_list}
        active, inactive = set(), set()
        for dot, point in cell['grid']['slots'].items():
            center = tuple(map(round, point))
            self.assertIn(center, rings)
            (active if dot in cell['dots'] else inactive).add(rings[center])
        self.assertEqual(len(active), 1)
        self.assertEqual(len(inactive), 1)
        self.assertNotEqual(active, inactive)
        self.assertGreater(next(iter(active))[0], next(iter(inactive))[0])
        self.assertIn('C81', [call.args[2] for call in text.call_args_list])

    def test_details_keeps_all_six_dot_numbers(self):
        with patch('yolo_detector.cv2.putText', wraps=cv2.putText) as text:
            self.draw(details=True)
        labels = [call.args[1] for call in text.call_args_list]
        for dot in range(1, 7):
            self.assertEqual(labels.count(str(dot)), len(self.cells))

    def test_grid_has_three_separators_following_skewed_slots_inside_bbox(self):
        cell = deepcopy(self.cells[0])
        cell['grid']['slots'] = {1: (30, 50), 2: (33, 70), 3: (36, 90),
                                4: (50, 54), 5: (53, 74), 6: (56, 94)}
        cell['grid']['bbox'] = (20, 40, 66, 104)
        with patch('yolo_detector.cv2.line', wraps=cv2.line) as line:
            self.reader._draw_cell_hud(np.zeros_like(self.image), [cell])
        self.assertEqual(line.call_count, 3)
        for call in line.call_args_list:
            a, b = call.args[1:3]
            self.assertNotEqual(a[0], b[0])
            self.assertNotEqual(a[1], b[1])
            for x, y in (a, b):
                self.assertTrue(20 <= x <= 66 and 40 <= y <= 104)

    def test_all_physical_cell_labels_survive_multicell_decoding_and_details(self):
        tokens = decode_cells_verbose(self.cells, 'thai')
        self.assertTrue(any(token.get('consumed') for token in tokens))
        for details in (False, True):
            self.reader._hud_label_cache = {}
            with self.subTest(details=details), \
                 patch.object(ImageDraw.ImageDraw, 'text', autospec=True,
                              side_effect=ImageDraw.ImageDraw.text) as text:
                self.reader.annotate_with_text(self.image, self.debug['dots'], self.cells,
                    verbose_results=tokens, details=details, footer=False, cell_hud=True)
                labels = [call.args[2] for call in text.call_args_list]
                self.assertEqual([value for value in labels if value.startswith('C')],
                                 [f'C{i+1}' for i in range(len(self.cells))])
                self.assertEqual(labels.count('ญ'), 1)
                self.assertIn('+', labels)

    def test_confirmed_thai_english_and_unknown_tokens_in_compact_preview(self):
        for lang, expected in [('thai', 'อ'), ('english', 'o')]:
            tokens = decode_cells_verbose(self.cells, lang)
            with self.subTest(lang=lang), \
                 patch.object(ImageDraw.ImageDraw, 'text', autospec=True,
                              side_effect=ImageDraw.ImageDraw.text) as text:
                self.reader.annotate_with_text(self.image, self.debug['dots'], self.cells,
                    verbose_results=tokens, details=False, footer=False, cell_hud=True)
                self.assertIn(expected, [call.args[2] for call in text.call_args_list])
        for token in (dict(char='ก', warning='ambiguous_row_grid'), dict(char='�')):
            self.reader._hud_label_cache = {}
            with patch.object(ImageDraw.ImageDraw, 'text', autospec=True,
                              side_effect=ImageDraw.ImageDraw.text) as text:
                self.reader._draw_cell_hud(np.zeros_like(self.image), self.cells[:1], [token])
                labels = [call.args[2] for call in text.call_args_list]
                self.assertIn('?', labels)
                self.assertNotIn('ก', labels)
                self.assertNotIn('�', labels)

    def test_new_inference_reuses_glyph_plates_and_cache_is_bounded(self):
        tokens = decode_cells_verbose(self.cells, 'thai')
        with patch.object(ImageDraw.ImageDraw, 'text', autospec=True,
                          side_effect=ImageDraw.ImageDraw.text) as text:
            first = np.zeros_like(self.image)
            self.reader._draw_cell_hud(first, self.cells, tokens)
            count = text.call_count
            second = np.zeros_like(self.image)
            self.reader._draw_cell_hud(second, self.cells, tokens)
            self.assertEqual(text.call_count, count)
            np.testing.assert_array_equal(first, second)
        self.reader._hud_label_cache = {('old', i): None for i in range(512)}
        header = self.reader._cell_hud_header('C900', 'ก', 20, 40, 720, False)
        self.assertEqual(len(self.reader._hud_label_cache), 1)
        self.assertGreater(header.size, 0)

    def test_header_and_pattern_have_no_solid_backing_on_light_or_dark_images(self):
        header = self.reader._cell_hud_header('C1', 'ก', 20, 50, 720, False)
        self.assertEqual(header.shape[2], 4)
        self.assertTrue(np.all(header[0, :, 3] == 0))
        self.assertGreater(np.count_nonzero(header[:, :, 3] == 0), header.shape[0]*header.shape[1]//3)
        self.assertTrue(np.any(np.all(header[:, :, :3] == (44, 31, 23), axis=2)))
        for shade in (30, 230):
            canvas = np.full_like(self.image, shade)
            with patch('yolo_detector.cv2.rectangle', wraps=cv2.rectangle) as rectangle:
                self.reader._draw_cell_hud(canvas, self.cells, decode_cells_verbose(self.cells, 'thai'))
            # Only cell outlines remain; neither labels nor patterns draw boxes.
            self.assertEqual(rectangle.call_count, len(self.cells))
            self.assertTrue(all(call.args[4] > 0 for call in rectangle.call_args_list))
            self.assertTrue(np.any(canvas == shade))

    def test_token_changes_invalidate_cached_labels_even_with_same_summary(self):
        preview = LivePreview(dashboard=True)
        result = dict(result_id=1, cells=self.cells, dots=self.debug['dots'],
                      decoded_text='', verbose_results=[])
        with patch.object(self.reader, 'annotate_with_text', wraps=self.reader.annotate_with_text) as draw:
            preview.render(self.image, self.reader, result, 'thai')
            result['verbose_results'] = decode_cells_verbose(self.cells, 'thai')
            confirmed = preview.render(self.image, self.reader, result, 'thai')
            preview.render(self.image, self.reader, result, 'thai')
            self.assertEqual(draw.call_count, 2)
            result['verbose_results'] = deepcopy(result['verbose_results'])
            result['verbose_results'][0]['warning'] = 'unknown_pattern'
            uncertain = preview.render(self.image, self.reader, result, 'thai')
            self.assertEqual(draw.call_count, 3)
            self.assertFalse(np.array_equal(confirmed, uncertain))

    def test_hud_does_not_modify_frame_geometry_dots_or_decoded_result(self):
        original_image, original_cells, original_dots = self.image.copy(), deepcopy(self.cells), deepcopy(self.debug['dots'])
        decoded = decode_cells(self.cells, 'thai')
        tokens = decode_cells_verbose(self.cells, 'thai')
        output = self.draw()
        self.assertEqual(output.shape, self.image.shape)
        np.testing.assert_array_equal(original_image, self.image)
        self.assertEqual(self.cells, original_cells)
        self.assertEqual(self.debug['dots'], original_dots)
        self.assertEqual(decoded, decode_cells(self.cells, 'thai'))
        self.assertEqual(tokens, decode_cells_verbose(self.cells, 'thai'))

    def test_resize_redraws_overlay_without_retracking_or_changing_cell_order(self):
        preview = LivePreview(max_width=800, dashboard=True)
        result = dict(result_id=1, cells=self.cells, dots=self.debug['dots'],
                      decoded_text='', verbose_results=[])
        with patch.object(preview._grids, 'update', wraps=preview._grids.update) as track, \
             patch.object(self.reader, 'annotate_with_text', wraps=self.reader.annotate_with_text) as draw:
            first = preview.render(self.image, self.reader, result, 'thai')
            preview.render(self.image, self.reader, result, 'thai')
            self.assertEqual(draw.call_count, 1)
            preview.max_width = 640
            second = preview.render(self.image, self.reader, result, 'thai')
            self.assertNotEqual(first.shape, second.shape)
            self.assertEqual(track.call_count, 1)
            self.assertEqual(draw.call_count, 2)
            self.assertEqual([c['dots'] for c in draw.call_args.args[2]], [c['dots'] for c in self.cells])

    def test_wrapped_pattern_stays_complete_at_bottom_edge_after_zoom(self):
        cell = deepcopy(self.cells[1])
        _, cells = scaled_geometry([], [cell], .4, .4)
        cell = cells[0]
        # A cell near the bottom of a normal preview still has room above it
        # for the label and wrapped pattern after the viewport is cropped.
        cell['grid']['slots'] = {dot: (x, y+100) for dot, (x, y) in cell['grid']['slots'].items()}
        cell['grid']['bbox'] = tuple(value+(100 if i % 2 else 0)
                                     for i, value in enumerate(cell['grid']['bbox']))
        height = round(cell['grid']['bbox'][3])+1
        canvas = np.zeros((height, 400, 3), np.uint8)
        with patch('yolo_detector.cv2.putText', wraps=cv2.putText) as text, \
             patch('yolo_detector.cv2.copyTo', wraps=cv2.copyTo) as copy:
            self.reader._draw_cell_hud(canvas, [cell])
        lines = [call for call in text.call_args_list if call.args[6] == 1
                 and ('[' in call.args[1] or ']' in call.args[1])]
        self.assertEqual(''.join(call.args[1] for call in lines), '[1,2,3,4,5,6]')
        self.assertEqual(len(lines), 2)
        # Pattern glyphs are drawn locally then copied into two distinct rows
        # above the header when there is no room below the cell.
        base_address = canvas.__array_interface__['data'][0]
        destinations = [call.args[2] for call in copy.call_args_list[:2]]
        rows = [(dst.__array_interface__['data'][0]-base_address)//canvas.strides[0]
                for dst in destinations]
        self.assertLess(rows[0], rows[1])
        self.assertLess(rows[1]+destinations[1].shape[0], cell['grid']['bbox'][1])

    def test_dense_multiline_patterns_do_not_overlap_horizontal_neighbors(self):
        # Render 54 physical cells with six-dot patterns in three rows; this
        # exercises wrapping at small cell pitch independently of a saved photo.
        prototype = self.cells[1]
        cells = []
        for line in range(3):
            for index in range(18):
                cell = deepcopy(prototype)
                origin = np.asarray(prototype['grid']['slots'][1])
                position = np.array([30+index*46, 40+line*100])
                cell['grid']['slots'] = {k: tuple((np.asarray(p)-origin)*.5+position)
                                        for k, p in prototype['grid']['slots'].items()}
                cell['grid']['bbox'] = tuple((value-origin[i % 2])*.5+position[i % 2]
                                            for i, value in enumerate(prototype['grid']['bbox']))
                cell['center'] = tuple((np.asarray(prototype['center'])-origin)*.5+position)
                cell['x'], cell['y'] = cell['center']
                cell['line_id'] = line
                cells.append(cell)
        # Scale the geometry so neighboring cells have genuinely dense lanes.
        _, cells = scaled_geometry([], cells, .5, .5)
        canvas = np.zeros((250, 500, 3), np.uint8)
        with patch('yolo_detector.cv2.putText', wraps=cv2.putText) as text:
            self.reader._draw_cell_hud(canvas, cells)
        patterns = [c for c in text.call_args_list if c.args[6] == 1
                    and ('[' in c.args[1] or ']' in c.args[1])]
        self.assertEqual(len(patterns), 108)
        for call in patterns:
            text_width = cv2.getTextSize(call.args[1], call.args[3], call.args[4], 1)[0][0]
            self.assertLessEqual(text_width+4, 23)


if __name__ == '__main__':
    unittest.main()
