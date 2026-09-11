"""Fault injection for frame recovery and camera/worker resource ownership."""

from collections import deque
from contextlib import ExitStack, redirect_stdout
import io
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner, ThreadedCameraCapture
from decoder import decode_cells
from yolo_cell_stream import UnreadableBrailleFrame, _line_rows


FRAME = np.zeros((120, 320, 3), np.uint8)


def await_condition(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.002)
    raise AssertionError('Timed out waiting for test condition')


class FakeCapture:
    def __init__(self, actions=()):
        self.actions = deque(actions)
        self.releases = 0
        self.reads = 0
        self.settings = []
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        self.reads += 1
        if self.actions:
            action = self.actions.popleft()
            if isinstance(action, Exception):
                raise action
            if callable(action):
                return action()
            return action
        time.sleep(.002)
        return True, FRAME.copy()

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 120,
                cv2.CAP_PROP_FPS: 60}.get(prop, 0)

    def release(self):
        self.releases += 1
        self.opened = False


def camera_with(cap):
    with patch.object(ThreadedCameraCapture, '_init_camera'):
        camera = ThreadedCameraCapture()
    camera.cap = cap
    return camera


class CaptureRecoveryTests(unittest.TestCase):
    def test_failed_backend_and_setup_exception_release_handles(self):
        first, second = FakeCapture(), FakeCapture()
        first.opened = False
        with patch('camera_reader.cv2.VideoCapture', side_effect=[first, second]):
            camera = ThreadedCameraCapture()
        self.assertEqual(first.releases, 1)
        self.assertEqual(second.reads, 0)  # no blocking first read on the UI thread
        camera.release()
        camera.release()
        self.assertEqual(second.releases, 1)
        cap = FakeCapture()
        with patch('camera_reader.cv2.VideoCapture', return_value=cap), \
                patch.object(cap, 'set', side_effect=RuntimeError('configuration failure')):
            with self.assertRaisesRegex(RuntimeError, 'configuration'):
                ThreadedCameraCapture()
        self.assertEqual(cap.releases, 1)

    def test_hundreds_of_bad_reads_and_exception_recover_on_same_handle(self):
        cap = FakeCapture([(False, None)]*320 +
                          [RuntimeError('native read failure'), (True, np.empty((0, 0, 3), np.uint8))])
        camera = camera_with(cap)
        with patch.object(camera, '_init_camera') as reopen, \
                patch.object(camera._stop_event, 'wait', side_effect=lambda delay: camera._stop_event.is_set()), \
                self.assertLogs('camera_reader', 'WARNING'):
            try:
                self.assertTrue(camera.start())
                original_thread = camera.thread
                self.assertTrue(camera.start())
                self.assertIs(camera.thread, original_thread)
                await_condition(lambda: camera.read_latest()[0])
                self.assertGreaterEqual(cap.reads, 323)
                reopen.assert_not_called()
            finally:
                camera.release()
        self.assertEqual(cap.releases, 1)

    def test_resolution_and_release_never_overlap_native_read(self):
        entered, resume, configured = threading.Event(), threading.Event(), threading.Event()
        reading = threading.Event()

        def blocked_read():
            reading.set()
            entered.set()
            if not resume.wait(3):
                raise TimeoutError('test read release')
            reading.clear()
            return True, FRAME.copy()

        cap = FakeCapture([blocked_read])
        original_set, original_release = cap.set, cap.release

        def checked_set(prop, value):
            self.assertFalse(reading.is_set())
            configured.set()
            return original_set(prop, value)

        def checked_release():
            self.assertFalse(reading.is_set())
            original_release()

        cap.set, cap.release = checked_set, checked_release
        camera = camera_with(cap)
        try:
            camera.start()
            self.assertTrue(entered.wait(2))
            camera.set_resolution(640, 480)
            self.assertFalse(configured.is_set())
            resume.set()
            self.assertTrue(configured.wait(2))
        finally:
            resume.set()
            camera.release()
        self.assertEqual(cap.releases, 1)

    def test_delayed_read_expires_frame_and_shutdown_keeps_owner(self):
        entered, resume = threading.Event(), threading.Event()

        def blocked_read():
            entered.set()
            resume.wait(3)
            return True, FRAME.copy()

        cap = FakeCapture([(True, FRAME.copy()), blocked_read])
        camera = camera_with(cap)
        try:
            camera.start()
            self.assertTrue(entered.wait(2))
            with camera.lock:
                camera._last_frame_at = time.monotonic() - camera.STALE_AFTER - 1
            self.assertFalse(camera.read_latest()[0])
            self.assertEqual(camera.get_status()[0], 'waiting')
            with self.assertLogs('camera_reader', 'WARNING'):
                self.assertFalse(camera.release(timeout=.01))
            self.assertEqual(cap.releases, 0)
            self.assertFalse(camera.start())
        finally:
            resume.set()
            camera.release()
        self.assertEqual(cap.releases, 1)
        self.assertFalse(camera.read_latest()[0])

    def test_persistent_capture_failure_has_bounded_reconnects(self):
        cap = FakeCapture([(False, None)])
        camera = camera_with(cap)
        camera.RETRY_AFTER, camera.MAX_RECONNECTS = 0, 2
        with patch.object(camera, '_init_camera', return_value=False) as reopen, \
                patch.object(camera._stop_event, 'wait', return_value=False), \
                self.assertLogs('camera_reader', 'WARNING'):
            camera.start()
            camera.thread.join(2)
        self.assertFalse(camera.thread.is_alive())
        self.assertEqual(reopen.call_count, 2)
        self.assertEqual(camera.get_status()[0], 'unavailable')
        camera.release()
        self.assertEqual(cap.releases, 1)


class WorkerRecoveryTests(unittest.TestCase):
    def result(self, worker, number):
        return await_condition(lambda: (r if (r := worker.get_latest_results())['result_id'] >= number else None))

    def test_unreadable_errors_and_uncertain_cells_then_success(self):
        mode = [0]

        class Detector:
            def detect(self, image):
                if mode[0] == 0:
                    _line_rows([[i] for i in range(5)], np.arange(5)*10., 10.)
                if mode[0] == 1:
                    raise RuntimeError('injected model error')
                if mode[0] == 2:
                    return [dict(dots=frozenset(), x=0, y=0, crop_status='empty')], {'dots': []}
                return [dict(dots=frozenset({1}), x=0, y=0)], {'dots': []}

        worker = AsyncBrailleWorker(Detector(), default_lang='english')
        worker.start()
        thread = worker.thread
        try:
            with self.assertLogs('camera_reader', 'ERROR'):
                for i in range(24):
                    mode[0] = i % 2
                    worker.submit_frame(FRAME, frame_id=i)
                    result = self.result(worker, i+1)
                    self.assertEqual(result['status'], 'unreadable' if mode[0] == 0 else 'error')
                    self.assertEqual(result['decoded_text'], '')
                    self.assertEqual(result['stage'], 'detection')
            mode[0] = 2
            worker.submit_frame(FRAME, frame_id=24)
            result = self.result(worker, 25)
            self.assertEqual(result['decoded_text'], '�')
            self.assertEqual(result['status'], 'uncertain')
            self.assertIn('empty_crop', result['reason'])
            mode[0] = 3
            worker.submit_frame(FRAME, frame_id=25)
            result = self.result(worker, 26)
            self.assertEqual((result['decoded_text'], result['status'], result['error']), ('a', 'ok', None))
            self.assertIs(worker.thread, thread)
            self.assertTrue(thread.is_alive())
        finally:
            worker.stop()

    def test_invalid_input_never_reaches_detector_and_decoder_exception_recovers(self):
        detector = Mock()
        detector.detect.return_value = ([dict(dots=frozenset({1}), x=0, y=0)], {'dots': []})
        worker = AsyncBrailleWorker(detector, default_lang='english')
        worker.start()
        try:
            for i, frame in enumerate([np.empty((0, 0, 3), np.uint8),
                                        np.zeros((5, 5), np.uint8), FRAME.astype(float)], 1):
                worker.submit_frame(frame)
                result = self.result(worker, i)
                self.assertEqual((result['status'], result['stage']), ('unreadable', 'preprocessing'))
            detector.detect.assert_not_called()
            with patch('camera_reader.cv2.GaussianBlur', side_effect=cv2.error('filter failure')), \
                    self.assertLogs('camera_reader', 'ERROR'):
                worker.submit_frame(FRAME, sharpness_strength=.35)
                result = self.result(worker, 4)
                self.assertEqual((result['status'], result['stage']), ('error', 'preprocessing'))
            with patch('camera_reader.decode_cells', side_effect=RuntimeError('decode failure')), \
                    self.assertLogs('camera_reader', 'ERROR'):
                worker.submit_frame(FRAME)
                self.assertEqual(self.result(worker, 5)['stage'], 'decoding')
            worker.submit_frame(FRAME)
            self.assertEqual(self.result(worker, 6)['decoded_text'], 'a')
        finally:
            worker.stop()

    def test_single_worker_owned_pending_frame_and_stop_during_inference(self):
        entered, resume = threading.Event(), threading.Event()
        seen = []

        class Detector:
            def detect(self, frame):
                seen.append(int(frame[0, 0, 0]))
                if len(seen) == 1:
                    entered.set()
                    resume.wait(3)
                return [], {'dots': []}

        worker = AsyncBrailleWorker(Detector())
        worker.start()
        thread = worker.thread
        try:
            self.assertTrue(worker.start())
            self.assertIs(worker.thread, thread)
            worker.submit_frame(FRAME, frame_id=1)
            self.assertTrue(entered.wait(2))
            pending = np.ones_like(FRAME)
            self.assertTrue(worker.submit_frame(pending, frame_id=2))
            self.assertFalse(worker.submit_frame(pending, frame_id=2))
            pending[:] = 9
            resume.set()
            self.result(worker, 2)
            self.assertEqual(seen, [0, 1])
        finally:
            resume.set()
            worker.stop()

        entered.clear()
        resume.clear()
        seen.clear()
        worker.start()
        worker.submit_frame(FRAME, frame_id=3)
        try:
            self.assertTrue(entered.wait(2))
            count = worker.get_latest_results()['result_id']
            with self.assertLogs('camera_reader', 'WARNING'):
                self.assertFalse(worker.stop(timeout=.01))
            self.assertFalse(worker.start())
            self.assertFalse(worker.submit_frame(FRAME, frame_id=4))
            resume.set()
            worker.thread.join(2)
            self.assertEqual(worker.get_latest_results()['result_id'], count)
            self.assertEqual(seen, [0])
        finally:
            resume.set()
            worker.stop()


class FakeSessionCamera:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.actual_width, self.actual_height, self.actual_fps = 320, 120, 60
        self.reads, self.releases, self.starts = 0, 0, 0
        self.thread = None

    def is_opened(self):
        return True

    def start(self):
        self.starts += 1
        return True

    def read_latest(self, with_id=False):
        self.reads += 1
        return next(self.frames)

    def get_status(self):
        return 'waiting', 'injected missing frame'

    def release(self):
        self.releases += 1
        return True


class SessionRecoveryTests(unittest.TestCase):
    def scanner(self):
        with patch('camera_reader.YOLOBrailleDetector'):
            return RealTimeBrailleScanner(res_preset='480p', initial_zoom=1)

    def session_patches(self, camera, keys):
        stack = ExitStack()
        self.addCleanup(stack.close)
        create = stack.enter_context(patch('camera_reader.ThreadedCameraCapture', return_value=camera))
        worker = stack.enter_context(patch('camera_reader.AsyncBrailleWorker'))
        for name in ('namedWindow', 'resizeWindow', 'setMouseCallback', 'destroyAllWindows', 'imshow'):
            stack.enter_context(patch('camera_reader.cv2.'+name))
        poll = stack.enter_context(patch('camera_reader.cv2.waitKey', side_effect=keys))
        stack.enter_context(patch('camera_reader.cv2.getWindowProperty', return_value=1))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return create, worker, poll

    def test_more_than_300_missing_frames_keep_ui_live_and_recover(self):
        camera = FakeSessionCamera([(False, None, 0)]*320 + [(True, FRAME, 1)])
        create, worker, poll = self.session_patches(camera, [255]*320 + [ord('q')])
        scanner = self.scanner()
        with patch.object(scanner, '_render_frame', return_value=(FRAME.copy(), FRAME)) as render:
            scanner.run()
        self.assertEqual(poll.call_count, 321)
        self.assertEqual(render.call_count, 1)
        self.assertEqual((create.call_count, camera.starts, camera.releases), (1, 1, 1))
        worker.return_value.stop.assert_called_once()

    def test_preview_and_snapshot_exception_do_not_end_session(self):
        camera = FakeSessionCamera([(True, FRAME, i) for i in range(4)])
        _, worker, poll = self.session_patches(camera, [255, ord('p'), 255, ord('q')])
        scanner = self.scanner()
        with patch.object(scanner, '_render_frame',
                          side_effect=[ValueError('broken overlay')] + [(FRAME.copy(), FRAME)]*3), \
                patch('camera_reader.os.makedirs', side_effect=OSError('storage unavailable')), \
                self.assertLogs('camera_reader', 'ERROR'):
            scanner.run()
        self.assertEqual(poll.call_count, 4)
        self.assertEqual(camera.releases, 1)
        worker.return_value.stop.assert_called_once()

    def test_diagnostic_serialization_failure_does_not_close_session(self):
        camera = FakeSessionCamera([(True, FRAME, 1), (True, FRAME, 2)])
        _, worker, poll = self.session_patches(camera, [ord('d'), ord('q')])
        worker.return_value.get_latest_results.return_value = dict(
            inference_frame=FRAME, camera_roi=FRAME, cells=[], debug_info={}, lang='thai',
            frame_id=1, result_id=1, status='empty', reason='no cells', stage='detection',
            error=None, context=('thai',), sharpness_strength=0, decoded_text='')
        scanner = self.scanner()
        with patch.object(scanner, '_render_frame', return_value=(FRAME.copy(), FRAME)), \
                patch('tools.diagnostics.export_yolo_stream.export_stream',
                      side_effect=TypeError('diagnostic metadata cannot be serialized')), \
                self.assertLogs('camera_reader', 'ERROR'):
            scanner.run()
        self.assertEqual(poll.call_count, 2)
        self.assertEqual(camera.releases, 1)
        worker.return_value.stop.assert_called_once()

    def test_setup_failure_cleans_both_owners(self):
        camera = FakeSessionCamera([])
        _, worker, _ = self.session_patches(camera, [])
        scanner = self.scanner()
        with patch('camera_reader.cv2.namedWindow', side_effect=cv2.error('window failure')):
            with self.assertRaises(cv2.error):
                scanner.run()
        self.assertEqual(camera.releases, 1)
        worker.return_value.stop.assert_called_once()

    def test_window_close_exits_even_without_camera_frame(self):
        camera = FakeSessionCamera([(False, None, 0)])
        self.session_patches(camera, [255])
        with patch('camera_reader.cv2.getWindowProperty', return_value=0):
            self.scanner().run()
        self.assertEqual(camera.releases, 1)

    def test_zoom_and_viewfinder_fit_tiny_frames(self):
        scanner = self.scanner()
        scanner.zoom_level = 4
        for shape in [(1, 1, 3), (30, 40, 3), (80, 120, 3), (720, 1280, 3)]:
            frame = np.zeros(shape, np.uint8)
            zoomed, box = scanner._apply_zoom(frame)
            self.assertGreater(zoomed.size, 0)
            scanner._draw_mini_viewfinder(frame, np.zeros((320, 480, 3), np.uint8), box)

    def test_thai_hud_uses_unicode_font_and_question_mark_keeps_meaning(self):
        scanner = self.scanner()
        # Use the same real font loader as the model, without loading weights.
        from yolo_detector import YOLOBrailleDetector
        font_owner = object.__new__(YOLOBrailleDetector)
        font_owner._font_cache = {}
        scanner.detector._get_font = font_owner._get_font
        with patch('camera_reader.cv2.putText', wraps=cv2.putText) as put:
            scanner._draw_top_hud(np.zeros((100, 1600, 3), np.uint8), 'ญ', True)
        self.assertTrue(all(call.args[1].isascii() for call in put.call_args_list))
        self.assertEqual(decode_cells([dict(dots=frozenset({2, 3, 6}), x=0, y=0)], 'english'), '?')



    def test_complete_flow_keeps_one_session_through_model_errors_and_recovery(self):
        notices, outputs = [], []

        class Detector:
            mode = 'yolo'

            def detect(self, frame, lang='english'):
                value = int(frame[0, 0, 0])
                if value == 1:
                    raise UnreadableBrailleFrame('ambiguous row grid')
                if value == 2:
                    raise RuntimeError('injected inference failure')
                return [dict(dots=frozenset({1}), center=(20, 20), x=20, y=20)], {'dots': []}

            def annotate_with_text(self, image, dots, cells, **options):
                outputs.append(options['decoded_text'])
                return image.copy()

        values = [1, 1, 2, 2, 1, 2, 0, 0, 0]
        camera = FakeSessionCamera([(True, np.full_like(FRAME, value), i+1)
                                    for i, value in enumerate(values)])
        scanner = self.scanner()
        scanner.detector, scanner.lang = Detector(), 'english'
        worker_threads = []

        def poll(delay):
            result = await_condition(lambda: (r if
                (r := scanner.ai_worker.get_latest_results())['frame_id'] == camera.reads else None))
            worker_threads.append(scanner.ai_worker.thread)
            return ord('q') if camera.reads == len(values) else 255

        with ExitStack() as stack:
            create = stack.enter_context(patch('camera_reader.ThreadedCameraCapture', return_value=camera))
            for name in ('namedWindow', 'resizeWindow', 'setMouseCallback', 'destroyAllWindows', 'imshow'):
                stack.enter_context(patch('camera_reader.cv2.'+name))
            stack.enter_context(patch('camera_reader.cv2.getWindowProperty', return_value=1))
            stack.enter_context(patch('camera_reader.cv2.waitKey', side_effect=poll))
            stack.enter_context(patch.object(scanner, '_draw_notice',
                                             side_effect=lambda frame, text: notices.append(text)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(self.assertLogs('camera_reader', 'WARNING'))
            scanner.run()
        self.assertEqual(create.call_count, 1)
        self.assertEqual((camera.starts, camera.releases, camera.reads), (1, 1, len(values)))
        self.assertEqual(len(set(worker_threads)), 1)
        self.assertIn('a', outputs)
        self.assertTrue(any('UNREADABLE' in text for text in notices))
        self.assertTrue(any('SCAN ERROR' in text for text in notices))
        self.assertFalse(scanner.ai_worker.thread.is_alive())

    def test_bad_render_result_is_quarantined_until_next_result(self):
        scanner = self.scanner()
        scanner.lang = 'english'
        scanner.ai_worker = Mock()
        frame_id = 1
        result = dict(ai_fps=1., result_id=1, lang='english', status='ok', reason=None, error=None,
                      source_shape=FRAME.shape, decoded_text='a', dots=[], verbose_results=[],
                      cells=[dict(dots=frozenset({1}), x=20, y=20)])  # missing center

        def latest():
            return dict(result, context=scanner._last_submitted_context)

        scanner.ai_worker.get_latest_results.side_effect = latest
        scanner.detector.mode = 'yolo'
        scanner.detector.annotate_with_text.side_effect = lambda image, *args, **kwargs: image.copy()
        with self.assertRaises(KeyError):
            scanner._render_frame(FRAME, frame_id)
        # Same result is suppressed without repeating the geometry exception or submission.
        scanner._render_frame(FRAME, frame_id)
        self.assertEqual(scanner.ai_worker.submit_frame.call_count, 1)
        result['result_id'] = 2
        result['cells'][0]['center'] = (20, 20)
        scanner._render_frame(FRAME, 2)
        self.assertEqual(scanner.history[-1], 'a')

    def test_old_context_and_slow_result_cannot_lock_current_text(self):
        scanner = self.scanner()
        scanner.lang = 'english'
        scanner.ai_worker = Mock()
        scanner.detector.mode = 'yolo'
        scanner.detector.annotate_with_text.side_effect = lambda image, *args, **kwargs: image.copy()
        result = dict(ai_fps=1., result_id=1, lang='english', status='ok', reason=None, error=None,
                      context=('old',), source_shape=FRAME.shape, decoded_text='a',
                      cells=[], dots=[], verbose_results=[])
        scanner.ai_worker.get_latest_results.return_value = result
        scanner.history.extend(['a']*scanner.stability_threshold)
        scanner._render_frame(FRAME, 1)
        self.assertFalse(scanner.history)
        result.update(context=scanner._last_submitted_context, busy_seconds=6)
        with patch.object(scanner, '_draw_notice') as notice:
            scanner._render_frame(FRAME, 2)
        self.assertFalse(scanner.history)
        self.assertIn('AI BUSY', notice.call_args.args[1])

    def test_pending_shutdown_cannot_create_a_second_session(self):
        scanner = self.scanner()
        scanner.ai_worker = Mock()
        scanner.ai_worker.thread.is_alive.return_value = True
        with patch('camera_reader.ThreadedCameraCapture') as create:
            with self.assertRaisesRegex(RuntimeError, 'shutdown'):
                scanner.run()
        create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
