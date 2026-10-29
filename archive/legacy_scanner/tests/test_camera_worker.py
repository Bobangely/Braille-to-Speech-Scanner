import threading
import time
import unittest

import numpy as np

from camera_reader import AsyncBrailleWorker


class CameraWorkerTests(unittest.TestCase):
    def await_result(self, worker, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = worker.get_latest_results()
            if predicate(result):
                return result
            time.sleep(.005)
        self.fail('Worker did not publish the expected result')

    def test_latest_pending_frame_retains_its_language_and_context(self):
        started, release = threading.Event(), threading.Event()
        calls = []

        class Detector:
            def detect(self, image, lang='thai'):
                value = int(image[0, 0, 0])
                calls.append((value, lang))
                if value == 1:
                    started.set()
                    if not release.wait(3):
                        raise TimeoutError('test release')
                return [], dict(dots=[], method='yolo_cell_stream', crop_count=0)

        worker = AsyncBrailleWorker(Detector())
        worker.start()
        try:
            worker.submit_frame(np.full((20, 20, 3), 1, np.uint8), lang='thai', context=('first',))
            self.assertTrue(started.wait(2))
            worker.submit_frame(np.full((20, 20, 3), 2, np.uint8), lang='thai', context=('skip',))
            worker.submit_frame(np.full((20, 20, 3), 3, np.uint8), lang='english', context=('latest',))
            release.set()
            result = self.await_result(worker, lambda r: r['context'] == ('latest',))
            self.assertEqual(calls, [(1, 'thai'), (3, 'english')])
            self.assertEqual(result['lang'], 'english')
            self.assertEqual(result['debug_info']['method'], 'yolo_cell_stream')
        finally:
            release.set()
            worker.stop()

    def test_image_only_detector_and_error_clear_old_results(self):
        class Detector:
            def detect(self, image):
                if image[0, 0, 0]:
                    raise RuntimeError('crop budget exceeded')
                return [dict(dots=frozenset({1}), x=0, y=0)], {'dots': []}

        worker = AsyncBrailleWorker(Detector(), default_lang='english')
        worker.start()
        try:
            worker.submit_frame(np.zeros((20, 20, 3), np.uint8))
            result = self.await_result(worker, lambda r: r['result_id'] == 1)
            self.assertEqual(result['decoded_text'], 'a')
            worker.submit_frame(np.ones((20, 20, 3), np.uint8))
            result = self.await_result(worker, lambda r: r['result_id'] == 2)
            self.assertEqual(result['error'], 'crop budget exceeded')
            self.assertEqual(result['cells'], [])
            self.assertEqual(result['decoded_text'], '')
            self.assertIsNone(result['source_shape'])
        finally:
            worker.stop()


if __name__ == '__main__':
    unittest.main()
