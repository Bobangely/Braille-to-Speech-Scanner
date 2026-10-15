"""เมนูและการตรวจ environment สำหรับ main.bat โดยไม่เปลี่ยน logic ของแอป"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
APP_MODULES = ('cv2', 'numpy', 'PIL', 'yaml', 'ultralytics', 'pythainlp')
SAMPLES = (
    ('Test_thai_01.png', 'thai'),
    ('test_thai_home.png', 'thai-legacy'),
    ('test_thai_cat.png', 'thai-legacy'),
    ('test_hello_blue.png', 'english'),
)


def run(command):
    result = subprocess.run([str(item) for item in command], cwd=ROOT)
    if result.returncode:
        print(f'[ERROR] Command failed with exit code {result.returncode}.', flush=True)
    return result.returncode


def check_modules(python, names, remedy):
    code = ('import importlib.util,sys; '
            f'missing=[name for name in {list(names)!r} if importlib.util.find_spec(name) is None]; '
            f'print("Missing dependencies: "+", ".join(missing)+". {remedy}" if missing else "",end="\\n" if missing else ""); '
            'sys.exit(bool(missing))')
    return run([python, '-X', 'utf8', '-c', code])


def application(script, args):
    target = ROOT/script
    if not target.is_file():
        raise FileNotFoundError(f'Missing application script: {target}')
    if code := check_modules(sys.executable, APP_MODULES, 'Run main.bat setup'):
        return code
    return run([sys.executable, '-X', 'utf8', target, *args])


def setup():
    requirements = ROOT/'requirements.txt'
    if not requirements.is_file():
        raise FileNotFoundError(f'Missing requirements file: {requirements}')
    env = ROOT/'.venv'
    python = env/'Scripts/python.exe'
    if env.exists() and not python.is_file():
        raise ValueError('The .venv folder is incomplete. Repair it manually; no environment was overwritten.')
    if not env.exists():
        if code := run([sys.executable, '-m', 'venv', env]):
            return code
    if code := run([python, '-m', 'pip', 'install', '--upgrade', 'pip']):
        return code
    if code := run([python, '-m', 'pip', 'install', '-r', requirements]):
        return code
    print('Environment setup complete. Run main.bat to open the menu.')
    return 0


def image(path, args, lang='thai', dump_stream=True):
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = Path(os.environ.get('BRAILLE_CALLER_CWD', ROOT))/source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Missing image: {source}')
    options = [source, '--mode', 'yolo', '--yolo-pipeline', 'stream', '--lang', lang, '--save']
    if dump_stream:
        options.append('--dump-stream')
    return application('yolo_detector.py', options+args)


def image_menu():
    result = 0
    while True:
        print('\nImage tests:')
        for index, (name, language) in enumerate(SAMPLES, 1):
            print(f'  {index}. {name} [{language}]')
        print('  5. Enter image path\n  Q. Back')
        choice = input('Select: ').strip().lower()
        if choice == 'q':
            return result
        try:
            if choice in ('1', '2', '3', '4'):
                name, language = SAMPLES[int(choice)-1]
                result = image(ROOT/'sample_images'/name, [], language, dump_stream=False)
            elif choice == '5':
                path = input('Image path: ').strip().strip('"')
                if path:
                    result = image(path, [])
            else:
                print('Choose 1-5 or Q.')
        except (OSError, ValueError) as exc:
            print(f'[ERROR] {exc}')
            result = 1


def scan(args, piper=False):
    options = ['--camera', '0', '--detector', 'yolo', '--yolo-pipeline', 'stream',
               '--res', 'auto', '--sharp', '0', '--lang', 'thai']
    if piper:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--tts-python', default=str(ROOT/'scratch/piper-env/Scripts/python.exe'))
        parser.add_argument('--tts-thai-model', default=str(ROOT/'scratch/piper-voices/th_TH-tsync2-medium.onnx'))
        parser.add_argument('--tts-english-model', default=str(ROOT/'scratch/piper-voices/en_US-ljspeech-medium.onnx'))
        voices, args = parser.parse_known_args(args)
        files = [voices.tts_python, voices.tts_thai_model, voices.tts_thai_model+'.json',
                 voices.tts_english_model, voices.tts_english_model+'.json']
        missing = [name for name in files if not Path(name).is_file()]
        if missing:
            raise FileNotFoundError('Missing local Piper files: '+', '.join(missing)+
                                    '. See docs/PIPER_OFFLINE_TTS_TH.md; no download or cloud fallback was attempted.')
        if code := check_modules(voices.tts_python, ('piper', 'onnxruntime', 'tltk', 'requests'),
                                 'See requirements-piper.txt and docs/PIPER_OFFLINE_TTS_TH.md'):
            return code
        options += ['--tts-python', voices.tts_python, '--tts-thai-model', voices.tts_thai_model,
                    '--tts-english-model', voices.tts_english_model]
        print('Offline Piper enabled. S: read/repeat; B: stop. Thai voice: non-commercial trial only.')
    else:
        print('Scanner without speech. Choose main.bat piper for offline Thai/English speech.')
    return application('camera_reader.py', options+args)


def dispatch(command, args):
    if command == 'scan':
        return scan(args)
    if command == 'piper':
        return scan(args, piper=True)
    if command == 'camera-test':
        return application('tools/diagnostics/check_camera.py', args)
    if command == 'image':
        return image(args[0], args[1:]) if args else image_menu()
    if command == 'setup':
        if args:
            raise ValueError('Usage: main.bat setup')
        return setup()
    raise ValueError(f'Unknown command: {command}. Run main.bat --help.')


def menu():
    result = 0
    choices = {'1': 'scan', '2': 'piper', '3': 'camera-test', '4': 'image', '5': 'setup'}
    while True:
        print('\nBraille Reader\n  1. Open scanner\n  2. Open scanner + offline Piper\n'
              '  3. Camera diagnostic\n  4. Image tests / custom image\n'
              '  5. Install project dependencies into .venv\n  Q. Exit')
        choice = input('Select: ').strip().lower()
        if choice == 'q':
            return result
        if choice not in choices:
            print('Choose 1-5 or Q.')
            continue
        try:
            result = dispatch(choices[choice], [])
        except (OSError, ValueError) as exc:
            print(f'[ERROR] {exc}')
            result = 1


def main():
    args = sys.argv[1:]
    if args and args[0] in ('--help', '-h'):
        print('main.bat                     Open menu\n'
              'main.bat scan [options]      Camera scanner\n'
              'main.bat piper [options]     Scanner + local/offline speech\n'
              'main.bat camera-test         Camera diagnostic\n'
              'main.bat image [path] [...]  Image menu or custom image\n'
              'main.bat "image path.png"     Drag/drop image also supported\n'
              'main.bat setup               Install dependencies into .venv\n'
              'Scanner/image options are forwarded; e.g. main.bat scan --res 1080p')
        return 0
    try:
        if sys.version_info < (3, 12):
            raise ValueError('Python 3.12 or newer is required; select a compatible project environment.')
        if not args:
            return menu()
        if args[0].startswith('-'):
            return scan(args)
        if Path(args[0]).suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp') or Path(args[0]).is_file():
            return image(args[0], args[1:])
        return dispatch(args[0], args[1:])
    except (OSError, ValueError) as exc:
        print(f'[ERROR] {exc}', flush=True)
        return 1
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
