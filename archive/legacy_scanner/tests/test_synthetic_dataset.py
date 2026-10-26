import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from tools.training.generate_yolo_training import generate_dataset
from tools.training.synthetic_braille import corpus, make_scene, render_variant
from tools.training.validate_dataset import validate_dataset
from tools.training import train_yolo


class SyntheticDatasetTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def generate(self, name='base', **options):
        defaults = dict(train_groups=2, val_groups=1, test_groups=1, variants=2, seed=111)
        defaults.update(options)
        with patch('builtins.print'):
            return generate_dataset(self.root/name, **defaults)

    def test_two_cell_symbols_and_tones_have_explicit_patterns(self):
        entries = dict(corpus())
        self.assertEqual(entries['ญ'], [[6], [1, 3, 4, 5, 6]])
        self.assertEqual(entries['ศ'], [[6], [2, 3, 4]])
        self.assertEqual(entries['ภ'], [[6], [1, 4, 5, 6]])
        self.assertEqual(entries['ธ'], [[3, 5, 6], [2, 3, 4, 5, 6]])
        self.assertEqual(entries['ก้า'], [[1, 2, 4, 5], [2, 5, 6], [1, 6]])

    def test_transformed_labels_cover_actual_rendered_ink(self):
        scene = make_scene(654, 3)
        encoded, labels, meta = render_variant(scene, 789)
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(len(labels.splitlines()), len(scene['objects']))
        for dot in meta['dots']:
            x, y = np.rint(dot['center']).astype(int)
            # Tests image pixels independently from the bbox-writing formula.
            self.assertLess(float(image[y-1:y+2, x-1:x+2].mean()), 200)
            x1, y1, x2, y2 = dot['bbox']
            self.assertTrue(x1 <= dot['center'][0] <= x2 and y1 <= dot['center'][1] <= y2)

    def test_negatives_have_no_positive_labels(self):
        scene = make_scene(10, 19)
        encoded, labels, metadata = render_variant(scene, 11)
        self.assertGreater(len(encoded), 100)
        self.assertEqual(labels, '')
        self.assertEqual(metadata['dots'], [])

    def test_reproducible_images_and_labels_with_isolated_scene_groups(self):
        first, second = self.generate('a'), self.generate('b')
        a = [json.loads(line) for line in (first/'records.jsonl').read_text().splitlines()]
        b = [json.loads(line) for line in (second/'records.jsonl').read_text().splitlines()]
        self.assertEqual(a, b)
        self.assertEqual(validate_dataset(first)['images'], {'train': 4, 'val': 2, 'test': 2})
        groups = {}
        for record in a:
            groups.setdefault(record['group_id'], set()).add(record['split'])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_extension_only_adds_train_and_preserves_original_holdouts(self):
        first = self.generate()
        original = (first/'records.jsonl').read_bytes()
        second = self.generate('extended', seed=222, val_groups=0, test_groups=0, extend_from=first)
        self.assertEqual(validate_dataset(second)['images'], {'train': 8, 'val': 2, 'test': 2})
        for split in ('val', 'test'):
            self.assertEqual((first/f'{split}.txt').read_bytes(), (second/f'{split}.txt').read_bytes())
        self.assertEqual((first/'records.jsonl').read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.generate('duplicate', val_groups=0, test_groups=0, extend_from=first)
        with self.assertRaises(FileExistsError):
            self.generate()

    def test_modified_box_is_rejected_even_if_file_hash_is_replaced(self):
        root = self.generate()
        records = [json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
        label_path = root/records[0]['label']
        lines = label_path.read_text().splitlines()
        lines[0] = '0 0.5 0.5 0.1 0.1'
        label_path.write_text('\n'.join(lines)+'\n')
        records[0]['label_sha256'] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        (root/'records.jsonl').write_text('\n'.join(map(json.dumps, records))+'\n')
        with self.assertRaisesRegex(ValueError, 'transform'):
            validate_dataset(root)

    def test_candidate_training_preserves_source_weights_and_separates_runs(self):
        dataset = self.generate()
        source = self.root/'source.pt'
        source.write_bytes(b'original weights')
        save_dir = self.root/'runs/detect/candidate'
        fake = SimpleNamespace(task='detect', names={0: 'braille_dot'}, trainer=SimpleNamespace(save_dir=save_dir))
        def fake_train(**kwargs):
            self.assertEqual(kwargs['fliplr'], 0)
            self.assertEqual(kwargs['flipud'], 0)
            self.assertFalse(kwargs['exist_ok'])
            (save_dir/'weights').mkdir(parents=True)
            (save_dir/'weights/best.pt').write_bytes(b'candidate weights')
            (save_dir/'weights/last.pt').write_bytes(b'last weights')
        fake.train = fake_train
        with patch.object(train_yolo, 'ROOT', self.root), patch('ultralytics.YOLO', return_value=fake), patch('builtins.print'):
            best = train_yolo.train(dataset, weights=source, epochs=1, name='candidate')
            self.assertTrue(best.is_file())
            self.assertEqual(source.read_bytes(), b'original weights')
            with self.assertRaises(FileExistsError):
                train_yolo.train(dataset, weights=source, epochs=1, name='candidate')


if __name__ == '__main__':
    unittest.main()
