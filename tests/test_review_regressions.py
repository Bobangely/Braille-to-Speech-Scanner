"""Failure cases found while reviewing the YOLO branch before merge."""

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from decoder import decode_cells, decode_cells_verbose
from tools.training.generate_yolo_training import generate_dataset
from tools.training.train_yolo import train
from tools.training.validate_dataset import validate_dataset
from yolo_detector import YOLOBrailleDetector
from camera_reader import RealTimeBrailleScanner, ThreadedCameraCapture


def stream_cell(pattern, index, **extra):
    return dict(dots=frozenset(pattern), line_id=0, reading_x=index * 95,
                cell_pitch=95, x=index * 95, y=0, **extra)


class ReadingFailureTests(unittest.TestCase):
    def test_logical_cells_without_coordinates_keep_adjacent_indicators(self):
        english = [dict(dots=p) for p in ({3, 4, 5, 6}, {1}, {1, 2})]
        thai = [dict(dots=p) for p in ({6}, {1, 3, 4, 5, 6})]
        self.assertEqual(decode_cells(english, 'english'), '12')
        self.assertEqual(decode_cells(thai, 'thai'), 'ญ')

    def test_english_decimal_keeps_number_mode_until_a_letter_ends_it(self):
        patterns = [{3, 4, 5, 6}, {1}, {2, 5, 6}, {1, 2}, {1, 3}, {1}]
        cells = [stream_cell(pattern, index) for index, pattern in enumerate(patterns)]
        self.assertEqual(decode_cells(cells, 'english'), '1.2ka')
        self.assertEqual(''.join(t['char'] for t in decode_cells_verbose(cells, 'english')), '1.2ka')

    def test_failed_crop_remains_visible_in_both_languages(self):
        cells = [stream_cell({1, 2, 4, 5}, 0),
                 stream_cell({}, 1, crop_status='empty', overview_dots={6}),
                 stream_cell({1, 2, 4, 5}, 2)]
        self.assertEqual(decode_cells(cells, 'thai'), 'ก�ก')
        self.assertEqual(decode_cells(cells, 'english'), 'g�g')
        token = decode_cells_verbose(cells, 'thai')[1]
        self.assertEqual(token['char'], '�')
        self.assertEqual(token['warning'], 'empty_crop')

    def test_failed_crop_ends_english_number_mode(self):
        cells = [stream_cell({3, 4, 5, 6}, 0), stream_cell({1}, 1),
                 stream_cell({}, 2, crop_status='empty'), stream_cell({1}, 3)]
        self.assertEqual(decode_cells(cells, 'english'), '1�a')

    def test_real_blank_cell_still_separates_words(self):
        cells = [stream_cell({1, 2, 4, 5}, 0), stream_cell({}, 1),
                 stream_cell({1, 2, 4, 5}, 2)]
        self.assertEqual(decode_cells(cells, 'thai'), 'ก ก')
        self.assertEqual(decode_cells(cells, 'english'), 'g g')


class ModelContractTests(unittest.TestCase):
    def test_wrong_model_task_or_classes_are_rejected_at_load(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'candidate.pt'
            checkpoint.write_bytes(b'test double')
            for task, names in [('classify', {0: 'braille_dot'}),
                                ('detect', {0: 'person'}),
                                ('detect', {0: 'braille_dot', 1: 'other'})]:
                model = SimpleNamespace(task=task, names=names)
                with self.subTest(task=task, names=names), \
                        patch('ultralytics.YOLO', return_value=model), \
                        self.assertRaisesRegex(ValueError, 'braille_dot'):
                    YOLOBrailleDetector(model_path=checkpoint)


class CameraFailureTests(unittest.TestCase):
    def test_every_advertised_resolution_is_applied(self):
        with patch('camera_reader.YOLOBrailleDetector'):
            for preset, expected in [('2k', (2560, 1440)), ('480p', (640, 480)),
                                     ('fhd', (1920, 1080)), ('hd', (1280, 720))]:
                with self.subTest(preset=preset):
                    scanner = RealTimeBrailleScanner(res_preset=preset)
                    self.assertEqual((scanner.target_width, scanner.target_height), expected)

    def test_incomplete_or_invalid_custom_resolution_is_rejected(self):
        with patch('camera_reader.YOLOBrailleDetector'):
            for options in [dict(width=640), dict(width=-1, height=480),
                            dict(res_preset='unknown')]:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    RealTimeBrailleScanner(**options)

    def test_failed_capture_does_not_keep_publishing_last_good_frame(self):
        with patch.object(ThreadedCameraCapture, '_init_camera'):
            camera = ThreadedCameraCapture()
        camera.frame, camera.ret, camera.running = object(), True, True

        def failed_read():
            camera.running = False
            return False, None

        camera.cap = SimpleNamespace(isOpened=lambda: True, read=failed_read)
        with patch('camera_reader.time.sleep'):
            camera._capture_loop()
        self.assertEqual(camera.read_latest(with_id=True), (False, None, 0))

    def test_closed_capture_does_not_keep_publishing_last_good_frame(self):
        with patch.object(ThreadedCameraCapture, '_init_camera'):
            camera = ThreadedCameraCapture()
        camera.frame, camera.ret, camera.running = object(), True, True
        camera.cap = SimpleNamespace(isOpened=lambda: False)
        camera._capture_loop()
        self.assertEqual(camera.read_latest(with_id=True), (False, None, 0))


class TrainingInputTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.dataset = generate_dataset(self.root / 'dataset', train_groups=1,
                                            val_groups=1, test_groups=1,
                                            variants=1, seed=345)

    def test_training_cannot_audit_one_yaml_and_use_another(self):
        alternate = self.dataset / 'unchecked.yaml'
        alternate.write_text('train: unaudited.txt\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'data.yaml'):
            train(alternate, weights=self.root / 'unused.pt')

    def test_yaml_cannot_redirect_split_lists_outside_audited_dataset(self):
        config_path = self.dataset / 'data.yaml'
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        config['path'] = (self.root / 'different_dataset').as_posix()
        config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'path'):
            validate_dataset(self.dataset)


if __name__ == '__main__':
    unittest.main()
