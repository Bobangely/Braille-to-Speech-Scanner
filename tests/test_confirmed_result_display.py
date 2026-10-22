"""Display history must not flicker or relax recognition confirmation."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from camera_reader import RealTimeBrailleScanner


class ConfirmedResultDisplayTests(unittest.TestCase):
    def setUp(self):
        self.scanner = RealTimeBrailleScanner(res_preset='480p', initial_zoom=1,
                                             stability_threshold=6)
        self.frame = np.zeros((120, 320, 3), np.uint8)
        self.result = dict(ai_fps=10., result_id=0, lang='thai', status='ok', reason=None,
                           error=None, source_shape=self.frame.shape, decoded_text='เกิด',
                           cells=[], dots=[], verbose_results=[], result_age=0.)
        self.scanner.ai_worker = Mock()
        self.scanner.ai_worker.get_latest_results.side_effect = lambda **kw: dict(
            self.result, context=self.scanner._last_submitted_context)
        self.scanner._preview = Mock(motion_valid=True)
        self.scanner._preview.render.side_effect = lambda frame, *args: frame.copy()

    def render(self, *, new=True, **changes):
        if new:
            self.result['result_id'] += 1
        self.result.update(changes)
        before = deepcopy(self.result)
        with patch('camera_reader.logger.warning'):
            self.scanner._render_frame(self.frame, self.result['result_id'])
        self.assertEqual(self.result, before)
        return self.scanner._ui._state

    def confirm(self, text='เกิด'):
        for _ in range(6):
            self.render(decoded_text=text, status='ok', result_age=0., busy_seconds=0.)
        self.assertEqual(self.scanner._ui._state['text'], text)
        self.assertFalse(self.scanner._ui._state['result_held'])

    def test_no_unconfirmed_text_and_duplicate_result_does_not_confirm(self):
        self.render()
        for _ in range(10):
            state = self.render(new=False)
            self.assertEqual(state['text'], '')
            self.assertEqual(state['confirmed'], 1)
        for _ in range(4):
            self.assertEqual(self.render()['text'], '')
        self.assertEqual(self.render()['text'], 'เกิด')

    def test_hold_on_changed_uncertain_empty_stale_and_busy_results(self):
        self.confirm()
        for change in [dict(decoded_text='เกอด'), dict(status='uncertain', decoded_text='�'),
                       dict(status='empty', decoded_text=''), dict(status='unreadable'),
                       dict(status='ok', decoded_text='สวน', result_age=2.),
                       dict(result_age=0., busy_seconds=6.)]:
            with self.subTest(change=change):
                state = self.render(**change)
                self.assertEqual(state['text'], 'เกิด')
                self.assertTrue(state['result_held'])
                self.assertNotEqual(state['status'], 'Detected')
                # Cached historical text must never become live cell labels.
                displayed = self.scanner._preview.render.call_args.args[2]
                self.assertEqual(displayed['decoded_text'], '')
                self.assertEqual(displayed['verbose_results'], [])

    def test_new_answer_replaces_old_only_after_six_new_matching_results(self):
        self.confirm()
        for text in ['สวน', 'สว', 'สวน', 'สว', 'สวน']:
            self.assertEqual(self.render(decoded_text=text)['text'], 'เกิด')
        for _ in range(4):
            state = self.render(decoded_text='สวน')
            self.assertEqual(state['text'], 'เกิด')
            self.assertTrue(state['result_held'])
        state = self.render(decoded_text='สวน')
        self.assertEqual(state['text'], 'สวน')
        self.assertFalse(state['result_held'])
        self.assertEqual(state['confirmed'], 6)

    def test_motion_failure_never_publishes_new_answer(self):
        self.confirm()
        for _ in range(5):
            self.render(decoded_text='สวน')
        self.scanner._preview.motion_valid = False
        state = self.render(decoded_text='สวน')
        self.assertEqual(state['text'], 'เกิด')
        self.assertTrue(state['result_held'])
        self.assertFalse(self.scanner.history)
        self.scanner._preview.motion_valid = True
        self.confirm('สวน')

    def test_context_changes_clear_old_display(self):
        for change in ('language', 'color', 'zoom', 'generation', 'resolution'):
            with self.subTest(change=change):
                self.setUp()
                self.confirm()
                if change == 'language':
                    self.scanner.lang = self.result['lang'] = 'english'
                elif change == 'color':
                    self.scanner.dot_color = 'red'
                elif change == 'zoom':
                    self.scanner.zoom_level = 1.2
                elif change == 'resolution':
                    self.frame = np.zeros((240, 640, 3), np.uint8)
                else:
                    self.scanner._context_generation += 1
                state = self.render(status='uncertain', decoded_text='')
                self.assertEqual(state['text'], '')
                self.assertFalse(state['result_held'])

    def test_wrong_source_shape_cannot_publish(self):
        self.confirm()
        for _ in range(5):
            self.render(decoded_text='สวน')
        state = self.render(decoded_text='สวน', source_shape=(240, 640, 3))
        self.assertEqual(state['text'], 'เกิด')
        self.assertTrue(state['result_held'])
        self.assertEqual(state['confirmed'], 0)

    def test_same_text_in_new_context_still_needs_six_new_results(self):
        self.confirm()
        self.scanner._context_generation += 1
        for _ in range(5):
            self.assertEqual(self.render()['text'], '')
        self.assertEqual(self.render()['text'], 'เกิด')

    def test_held_label_and_scroll_position(self):
        self.confirm('\n'.join(f'{i} เกิด' for i in range(50)))
        self.scanner._ui.scroll(1000)
        offset = self.scanner._ui.scroll_offset
        with patch.object(self.scanner._ui, '_label', wraps=self.scanner._ui._label) as labels:
            self.render(status='uncertain')
        self.assertEqual(self.scanner._ui.scroll_offset, offset)
        self.assertTrue(any(call.args[2] == 'ผลล่าสุด · กำลังอ่านใหม่' for call in labels.call_args_list))


if __name__ == '__main__':
    unittest.main()
