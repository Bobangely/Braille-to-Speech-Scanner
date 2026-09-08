import unittest
from unittest.mock import Mock
import numpy as np

from live_preview import LivePreview, scaled_geometry


class PreviewTests(unittest.TestCase):
    def test_reuses_overlay_but_always_displays_fresh_camera_pixels(self):
        detector = Mock(mode='hybrid')
        def annotate(image, *args, **kwargs):
            image[0, 0] = (0, 255, 0)
            return image
        detector.annotate_with_text.side_effect = annotate
        renderer = LivePreview(max_width=320)
        result = dict(result_id=1, dots=[], cells=[], decoded_text='', verbose_results=[])
        first = np.full((600, 800, 3), 40, np.uint8)
        second = np.full_like(first, 180)
        renderer.render(first, detector, result, 'thai')
        output = renderer.render(second, detector, result, 'thai')
        self.assertEqual(detector.annotate_with_text.call_count, 1)
        self.assertEqual(output.shape, (240, 320, 3))
        np.testing.assert_array_equal(output[100, 100], [180, 180, 180])
        self.assertTrue(np.all(second == 180))
        renderer.render(second, detector, dict(result, result_id=2), 'thai')
        self.assertEqual(detector.annotate_with_text.call_count, 2)

    def test_geometry_scaling_preserves_input_and_dot_numbering(self):
        dots = [dict(center=(100, 80), area=100, bbox=(90, 70, 110, 90))]
        cells = [dict(center=(100, 100), x=100, y=100, dots={6},
                      grid=dict(expected_cols=[80, 120], expected_rows=[60, 100, 140],
                                slots={6: (120, 140)}, bbox=(70, 50, 130, 150)))]
        small_dots, small_cells = scaled_geometry(dots, cells, .5, .5)
        self.assertEqual(small_dots[0]['area'], 25)
        self.assertEqual(small_cells[0]['grid']['slots'][6], (60, 70))
        self.assertEqual(cells[0]['grid']['slots'][6], (120, 140))


if __name__ == '__main__':
    unittest.main()
