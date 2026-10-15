"""ทดสอบ main.bat ผ่าน cmd จริง โดยใช้แอปจำลอง ไม่เปิดกล้องหรือติดตั้งแพ็กเกจ"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'nt', 'main.bat requires Windows cmd')
class MainLauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'main.bat').is_file(), 'ยังไม่มี entry point main.bat')
        self.temp = tempfile.TemporaryDirectory(prefix='braille-launch-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'project & space!'
        self.root.mkdir()
        shutil.copy2(ROOT/'main.bat', self.root/'main.bat')
        (self.root/'tools').mkdir()
        shutil.copy2(ROOT/'tools/launch.py', self.root/'tools/launch.py')
        self.make_python(self.root/'.venv')
        for module in ('cv2', 'numpy', 'PIL', 'yaml', 'ultralytics', 'pythainlp',
                       'piper', 'onnxruntime', 'tltk', 'requests'):
            (self.root/(module+'.py')).write_text('')
        stub = ("import json,os,sys\n"
                "print('LAUNCH:'+json.dumps(dict(script=os.path.basename(__file__),"
                "args=sys.argv[1:],cwd=os.getcwd())))\n"
                "sys.exit(int(os.environ.get('BRAILLE_TEST_APP_EXIT','0')))\n")
        for relative in ('camera_reader.py', 'yolo_detector.py', 'tools/diagnostics/check_camera.py'):
            path = self.root/relative
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(stub)
        (self.root/'requirements.txt').write_text('test-only-requirement\n')
        (self.root/'pip.py').write_text(
            "import os,sys\nprint('PIPSTEP:'+str(sys.argv[1:]))\n"
            "sys.exit(int(os.environ.get('BRAILLE_TEST_PIP_EXIT','0')))\n")

    def make_python(self, env):
        (env/'Scripts').mkdir(parents=True)
        shutil.copy2(sys.executable, env/'Scripts/python.exe')
        (env/'pyvenv.cfg').write_text('home = '+sys.base_prefix+'\ninclude-system-site-packages = false\n')

    def run_bat(self, *args, cwd=None, input='\n', env=None):
        settings = dict(os.environ, PYTHONUTF8='1')
        if env:
            settings.update(env)
        cmd = Path(os.environ['SystemRoot'])/'System32/cmd.exe'
        arguments = ' '.join('"'+str(arg)+'"' for arg in args)
        command = f'"{cmd}" /d /s /c ""{self.root / "main.bat"}" {arguments}"'
        return subprocess.run(
            command,
            input=input, capture_output=True, encoding='utf-8', errors='replace',
            cwd=cwd or self.root, env=settings, timeout=15)

    def launch(self, result):
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return json.loads(next(line.split('LAUNCH:', 1)[1] for line in result.stdout.splitlines() if 'LAUNCH:' in line))

    def test_scan_defaults_and_overrides_work_from_another_directory(self):
        result = self.run_bat('scan', '--res', '1080p', '--camera', '2', cwd=self.temp.name)
        launched = self.launch(result)
        self.assertEqual(Path(launched['cwd']), self.root)
        self.assertEqual(launched['script'], 'camera_reader.py')
        self.assertEqual(launched['args'][-4:], ['--res', '1080p', '--camera', '2'])
        self.assertIn('auto', launched['args'])

    def test_double_click_menu_can_open_scanner_and_exit(self):
        launched = self.launch(self.run_bat(input='1\nq\n'))
        self.assertEqual(launched['script'], 'camera_reader.py')

    def test_camera_diagnostic_is_preserved(self):
        launched = self.launch(self.run_bat('camera-test'))
        self.assertEqual(launched['script'], 'check_camera.py')

    def test_dropped_image_with_unicode_spaces_and_ampersand_is_not_shell_code(self):
        image = Path(self.temp.name)/'ภาพ A & B (test).png'
        image.write_bytes(b'fixture')
        launched = self.launch(self.run_bat(image, '--no-show'))
        self.assertEqual(launched['script'], 'yolo_detector.py')
        self.assertEqual(Path(launched['args'][0]), image)
        self.assertIn('--dump-stream', launched['args'])
        self.assertEqual(launched['args'][-1], '--no-show')

    def test_relative_image_path_uses_callers_directory(self):
        image = Path(self.temp.name)/'relative image.png'
        image.write_bytes(b'fixture')
        launched = self.launch(self.run_bat('image', image.name, cwd=self.temp.name))
        self.assertEqual(Path(launched['args'][0]), image)

    def test_old_legacy_and_english_sample_menu_choices_are_preserved(self):
        folder = self.root/'sample_images'
        folder.mkdir()
        for name in ('test_thai_home.png', 'test_hello_blue.png'):
            (folder/name).write_bytes(b'fixture')
        for choice, language in [('2', 'thai-legacy'), ('4', 'english')]:
            launched = self.launch(self.run_bat('image', input=choice+'\nq\n'))
            i = launched['args'].index('--lang')
            self.assertEqual(launched['args'][i+1], language)
            self.assertNotIn('--dump-stream', launched['args'])

    def test_python_on_path_can_launch_when_project_env_is_absent(self):
        shutil.rmtree(self.root/'.venv')
        search_path = str(Path(sys.executable).parent)+';'+str(Path(os.environ['SystemRoot'])/'System32')
        launched = self.launch(self.run_bat('scan', env={'PATH': search_path}))
        self.assertEqual(launched['script'], 'camera_reader.py')

    def test_missing_dependency_does_not_install_anything(self):
        (self.root/'cv2.py').unlink()
        result = self.run_bat('scan')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cv2', result.stdout+result.stderr)
        self.assertIn('main.bat setup', result.stdout+result.stderr)
        self.assertNotIn('PIPSTEP:', result.stdout)
        self.assertNotIn('LAUNCH:', result.stdout)

    def test_missing_script_and_image_produce_nonzero_exit(self):
        (self.root/'camera_reader.py').unlink()
        for args in [('scan',), ('image', 'missing.png')]:
            result = self.run_bat(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing', (result.stdout+result.stderr).lower())

    def test_piper_requires_local_files_then_forwards_offline_configuration(self):
        missing = self.run_bat('piper')
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('Piper', missing.stdout+missing.stderr)
        self.make_python(self.root/'scratch/piper-env')
        voices = self.root/'scratch/piper-voices'
        voices.mkdir()
        for name in ('th_TH-tsync2-medium.onnx', 'en_US-ljspeech-medium.onnx'):
            for suffix in ('', '.json'):
                (voices/(name+suffix)).write_text('fixture')
        launched = self.launch(self.run_bat('piper'))
        self.assertIn('--tts-python', launched['args'])
        self.assertIn('--tts-thai-model', launched['args'])
        self.assertIn('--tts-english-model', launched['args'])

    def test_application_error_exit_code_is_preserved(self):
        result = self.run_bat('scan', env={'BRAILLE_TEST_APP_EXIT': '9'})
        self.assertEqual(result.returncode, 9, result.stdout+result.stderr)

    def test_setup_installs_only_into_project_env_and_propagates_pip_failure(self):
        result = self.run_bat('setup', env={'BRAILLE_TEST_PIP_EXIT': '7'})
        self.assertEqual(result.returncode, 7, result.stdout+result.stderr)
        self.assertEqual(result.stdout.count('PIPSTEP:'), 1)
        self.assertNotIn('complete', result.stdout.lower())

    def test_no_python_and_missing_helper_are_clear_errors(self):
        shutil.rmtree(self.root/'.venv')
        system32 = str(Path(os.environ['SystemRoot'])/'System32')
        result = self.run_bat('scan', env={'PATH': system32})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python', result.stdout+result.stderr)
        (self.root/'tools/launch.py').unlink()
        result = self.run_bat('scan')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('launch.py', result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
