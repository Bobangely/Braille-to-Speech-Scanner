import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.diagnostics.benchmark_thai_tts import SOURCES, download_models


class ModelDownloadTests(unittest.TestCase):
    def test_bad_download_never_overwrites_an_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'mms/model.safetensors'
            model.parent.mkdir()
            model.write_bytes(b'original')

            def retrieve(url, destination):
                Path(destination).write_bytes(b'corrupt')

            source = [('facebook/mms-tts-tha', 'a' * 40, 'mms', [('model.safetensors', '0' * 64)])]
            with patch('tools.diagnostics.benchmark_thai_tts.SOURCES', source), \
                    patch('urllib.request.urlretrieve', side_effect=retrieve):
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    download_models(root)
            self.assertEqual(model.read_bytes(), b'original')
            self.assertFalse(model.with_suffix('.safetensors.part').exists())

    def test_weight_downloads_are_pinned_and_hash_checked(self):
        for _, revision, _, files in SOURCES:
            self.assertRegex(revision, r'^[0-9a-f]{40}$')
            for filename, digest in files:
                if filename.endswith(('.onnx', '.safetensors')):
                    self.assertRegex(digest, r'^[0-9a-f]{64}$')
