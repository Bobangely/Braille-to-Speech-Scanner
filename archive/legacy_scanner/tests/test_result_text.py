"""Rendering regressions: correct Unicode must also have correctly placed marks."""

import sys
import unicodedata
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from result_text import ResultText
from scanner_ui import ScannerUI


@unittest.skipUnless(sys.platform == 'win32', 'Windows fallback for Pillow without RAQM')
class WindowsResultTextTests(unittest.TestCase):
    def setUp(self):
        self.renderer = ResultText()
        self.font = ImageFont.truetype('C:/Windows/Fonts/tahoma.ttf', 25,
                                       layout_engine=ImageFont.Layout.BASIC)

    def test_tone_is_above_short_i_not_superimposed_as_long_i(self):
        for size in (18, 25, 32):
            for consonant in 'วกนสย':
                with self.subTest(size=size, consonant=consonant):
                    tops = []
                    for marks in ('ิ', 'ี', 'ิ่'):
                        mask, pad = self.renderer._mask(consonant+marks+'ง', 'Tahoma', size)
                        self.assertIsNotNone(mask.getbbox())
                        tops.append(mask.getbbox()[1]-pad)
                    self.assertLess(tops[2], tops[0])
                    self.assertLess(tops[2], tops[1])

    def test_native_masks_and_measurements_are_cached(self):
        image = Image.new('RGB', (300, 80))
        draw = ImageDraw.Draw(image)
        for _ in range(10):
            self.renderer.width(self.font, 'วิ่ง')
            self.renderer.draw(image, draw, (10, 10), 'วิ่ง', self.font, 'white')
        self.assertEqual(self.renderer._measure.cache_info().misses, 1)
        self.assertEqual(self.renderer._mask.cache_info().misses, 1)
        self.assertEqual(self.renderer._mask.cache_info().hits, 9)

    def test_english_still_uses_pillow_unchanged(self):
        self.assertFalse(self.renderer.native(self.font, 'Hello & D12'))
        actual = Image.new('RGB', (300, 80))
        expected = actual.copy()
        self.renderer.draw(actual, ImageDraw.Draw(actual), (10, 10),
                           'Hello & D12', self.font, 'white')
        ImageDraw.Draw(expected).text((10, 10), 'Hello & D12', font=self.font, fill='white')
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(self.renderer.width(self.font, 'Hello & D12'), self.font.getlength('Hello & D12'))

    def test_raqm_font_does_not_use_native_fallback(self):
        with patch.object(self.font, 'layout_engine', ImageFont.Layout.RAQM):
            self.assertFalse(self.renderer.native(self.font, 'วิ่ง'))

    def test_result_resize_wrap_scroll_and_state_preserve_unicode(self):
        def loader(size, bold=False):
            return self.font.font_variant(size=size)
        ui = ScannerUI(loader)
        text = '\n'.join(f'{i}: คุณแม่ตกใจวิ่งหนีจนลืม กิ่ง นิ่ง สิ่ง ยิ่ง' for i in range(12))
        state = dict(text=text, lang='thai', confirmed=6, required=6)
        for width, height in ((800, 600), (1920, 1080), (3840, 2160), (900, 1200)):
            with self.subTest(size=(width, height)):
                ui.resize(width, height)
                ui.scroll(-1000)
                canvas = ui.compose(None, state)
                self.assertEqual(''.join(ui._lines), text.replace('\n', ''))
                font = ui._font(20)
                available = ui.result_card[2]-ui.result_card[0]-20*ui.scale
                for line in ui._lines:
                    self.assertLessEqual(ui._result_text.width(font, line), available)
                    self.assertFalse(unicodedata.category(line[0]).startswith('M'))
                self.assertGreater(ui.max_scroll, 0)
                ui.scroll(1000)
                scrolled = ui.compose(None, state)
                self.assertEqual(ui.scroll_offset, ui.max_scroll)
                self.assertFalse(np.array_equal(canvas, scrolled))
                self.assertEqual(state['text'], text)
                self.assertEqual(state['confirmed'], 6)


if __name__ == '__main__':
    unittest.main()
