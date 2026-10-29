"""Teacher-panel settings: validated as a whole, saved atomically, applied to speech."""
import json
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from panel_settings import DEFAULTS, apply_settings, load, save, validate


class PanelSettingsTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / 'sub' / 'panel_settings.json'

    def tearDown(self):
        self._dir.cleanup()

    def test_validate_partial_keeps_others(self):
        self.assertEqual(validate({'mode': 'word'}, DEFAULTS), {**DEFAULTS, 'mode': 'word'})

    def test_validate_rejects(self):
        bad = [{'mode': 'loud'}, {'spell_max_cells': 0}, {'spell_max_cells': 6}, {'spell_max_cells': True},
               {'spell_max_cells': '3'}, {'camera': 'yes'}, {'volume': 5}, ['mode']]
        for partial in bad:
            with self.subTest(partial=partial), self.assertRaises(ValueError):
                validate(partial, DEFAULTS)

    def test_validate_all_or_nothing(self):
        current = dict(DEFAULTS)
        with self.assertRaises(ValueError):
            validate({'mode': 'word', 'spell_max_cells': 9}, current)
        self.assertEqual(current, DEFAULTS)

    def test_load_missing_file(self):
        self.assertEqual(load(self.path), DEFAULTS)

    def test_load_broken_file(self):
        self.path.parent.mkdir(parents=True)
        for text in ('{', '[1]', '{"mode": "loud"}'):
            with self.subTest(text=text):
                self.path.write_text(text, encoding='utf-8')
                self.assertEqual(load(self.path), DEFAULTS)

    def test_load_fills_missing_keys(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{"mode": "word"}', encoding='utf-8')
        self.assertEqual(load(self.path), {**DEFAULTS, 'mode': 'word'})

    def test_save_then_load(self):
        settings = {'mode': 'word', 'spell_max_cells': 4, 'camera': False}
        save(self.path, settings)
        self.assertEqual(load(self.path), settings)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ['panel_settings.json'])

    def test_apply_settings(self):
        speech = SimpleNamespace(mode='spell', spell_max_cells=2)
        result = apply_settings(speech, self.path, {'mode': 'word', 'spell_max_cells': 3})
        self.assertEqual(result, {'mode': 'word', 'spell_max_cells': 3, 'camera': True})
        self.assertEqual((speech.mode, speech.spell_max_cells), ('word', 3))
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8')), result)

    def test_apply_save_failure_changes_nothing(self):
        speech = SimpleNamespace(mode='spell', spell_max_cells=2)
        with mock.patch('panel_settings.os.replace', side_effect=PermissionError('file in use')):
            with self.assertRaises(PermissionError):
                apply_settings(speech, self.path, {'mode': 'word'})
        self.assertEqual(speech.mode, 'spell')

    def test_apply_concurrent_keeps_every_change(self):
        speech = SimpleNamespace(mode='spell', spell_max_cells=2)
        changes = [{'mode': 'word'}, {'spell_max_cells': 4}, {'camera': False}] * 5
        threads = [threading.Thread(target=apply_settings, args=(speech, self.path, c)) for c in changes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(load(self.path), {'mode': 'word', 'spell_max_cells': 4, 'camera': False})

    def test_apply_invalid_changes_nothing(self):
        speech = SimpleNamespace(mode='spell', spell_max_cells=2)
        save(self.path, DEFAULTS)
        with self.assertRaises(ValueError):
            apply_settings(speech, self.path, {'mode': 'word', 'camera': 1})
        self.assertEqual((speech.mode, speech.spell_max_cells), ('spell', 2))
        self.assertEqual(load(self.path), DEFAULTS)


if __name__ == '__main__':
    unittest.main()
