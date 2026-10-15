"""Download the Thai Piper voice required by the Braille reader.

Run once from the project root:
    .venv\\Scripts\\python.exe tools\\setup_piper_thai.py
"""

from pathlib import Path
from urllib.request import urlretrieve


VOICE_NAME = 'th_TH-mms_female-medium.onnx'
VOICE_BASE_URL = (
    'https://huggingface.co/tranhuyluyen/piper-voices-thai/resolve/main/'
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / 'models' / 'piper'


def download(filename):
    destination = DESTINATION / filename
    if destination.is_file() and destination.stat().st_size > 0:
        print(f'Already installed: {destination}')
        return
    print(f'Downloading {filename} ...')
    urlretrieve(VOICE_BASE_URL + filename, destination)
    print(f'Installed: {destination}')


def main():
    try:
        import piper  # noqa: F401
    except ImportError as error:
        raise SystemExit('Install dependencies first: .venv\\Scripts\\python.exe -m pip install -r requirements.txt') from error

    DESTINATION.mkdir(parents=True, exist_ok=True)
    download(VOICE_NAME)
    download(f'{VOICE_NAME}.json')
    print('\nPiper Thai voice is ready for offline use.')


if __name__ == '__main__':
    main()
