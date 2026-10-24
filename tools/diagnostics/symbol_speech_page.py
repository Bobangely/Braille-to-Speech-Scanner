"""Write an HTML page that plays the reader's spoken name for every symbol.

Uses the same voice and WAV cache as finger_reader.py, so what you hear is
what the reader says. Open the page in a browser and note the wrong names.
Usage: symbol_speech_page.py --tts-thai-model voice.onnx --tts-python env/python.exe
"""

from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from finger_reader import CachedWavSpeaker  # noqa: E402
from speech_queue import UNKNOWN  # noqa: E402
from thai_speech import SYMBOL_NAMES, THAI_CONSONANT_NAMES, tts_spelling  # noqa: E402

PLAY_ALL = '''<script>
function playAll() {
  const players = [...document.querySelectorAll('audio')];
  players.forEach((p, i) => p.onended = () => players[i + 1] && players[i + 1].play());
  players[0].play();
}
</script>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tts-thai-model', type=Path, required=True)
    parser.add_argument('--tts-thai-backend', choices=['piper', 'mms'], default='piper')
    parser.add_argument('--tts-python', help='Python executable of the Piper/MMS environment')
    parser.add_argument('--cache', type=Path, default=ROOT / 'output' / 'finger_speech')
    parser.add_argument('--edge-python', help='also play the online Edge voice (edge-tts environment)')
    parser.add_argument('--symbol-cache', type=Path, default=ROOT / 'output' / 'symbol_voice_edge')
    parser.add_argument('--out', type=Path, default=ROOT / 'output' / 'symbol_speech')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    entries = [(s, n, 'พยัญชนะ') for s, n in THAI_CONSONANT_NAMES.items()]
    entries += [(s, n, 'สระ / วรรณยุกต์') for s, n in SYMBOL_NAMES.items()]
    entries.append(('?', UNKNOWN, 'อ่านไม่ออก'))
    names = [name for _, name, _ in entries]
    voices = [('Piper', CachedWavSpeaker(args.tts_thai_model, args.cache, args.tts_python, args.tts_thai_backend))]
    if args.edge_python:
        voices.append(('Edge', CachedWavSpeaker(None, args.symbol_cache, args.edge_python, 'edge')))
    ready = {label: speaker.prepare_many(names) for label, speaker in voices}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, (symbol, name, group) in enumerate(entries, 1):
        audio = ''
        for label, speaker in voices:
            cell = '<i>สังเคราะห์ไม่สำเร็จ</i>'
            if name in ready[label]:
                wav = f'{label.lower()}_{number:02d}.wav'
                shutil.copyfile(speaker._path(name), args.out / wav)
                cell = f'<audio controls preload="none" src="{wav}"></audio>'
            audio += f'<td>{cell}</td>'
        rows.append(f'<tr><td>{number}</td><td>{group}</td><td class="s">{html.escape(symbol)}</td>'
                    f'<td>{html.escape(name)}</td><td>{html.escape(tts_spelling(name))}</td>{audio}</tr>')
    page = args.out / 'index.html'
    page.write_text(
        '<!doctype html><meta charset="utf-8"><title>เสียงชื่อสัญลักษณ์</title>'
        '<style>body{font-family:Tahoma,sans-serif;margin:16px}td,th{padding:4px 10px;border-bottom:1px solid #ddd}'
        '.s{font-size:24px}</style>'
        f'<h1>เสียงชื่อสัญลักษณ์ ({len(entries)} รายการ)</h1>'
        '<p>เสียงเดียวกับที่ตัวอ่านพูด จดเลขแถวที่ออกเสียงผิด</p>'
        '<button onclick="playAll()">เล่นทั้งหมดต่อกัน</button>'
        '<table><tr><th>#</th><th>กลุ่ม</th><th>สัญลักษณ์</th><th>ชื่อ</th><th>ข้อความที่ส่งให้ Piper</th>'
        + ''.join(f'<th>{label}</th>' for label, _ in voices) + '</tr>' + ''.join(rows) + '</table>' + PLAY_ALL, encoding='utf-8')
    print(page, ', '.join(f'{label} {len(r)}/{len(entries)}' for label, r in ready.items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
