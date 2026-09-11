"""Export the exact crop inputs and per-cell decisions for a YOLO stream run."""
import json
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from decoder import decode_cells, decode_cells_verbose
from yolo_cell_stream import crop_cell


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def text_trace(text):
    """Record the actual string before rendering; no replacement or guessed text."""
    return dict(text=text, codepoints=[f'U+{ord(char):04X}' for char in text],
                utf8_hex=text.encode('utf-8').hex(),
                replacement_characters=text.count('\ufffd'), question_marks=text.count('?'))


def export_stream(image, cells, debug_info, output_dir, lang='thai', source=None,
                  captured_image=None, metadata=None, detector=None):
    if debug_info.get('method') != 'yolo_cell_stream':
        raise ValueError('Crop export requires --mode yolo --yolo-pipeline stream')
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination/'inference_input.png'), image):
        raise OSError('Cannot save exact inference input')
    if captured_image is not None and not cv2.imwrite(str(destination/'camera_roi.png'), captured_image):
        raise OSError('Cannot save camera ROI before preprocessing')
    records = []
    tokens = decode_cells_verbose(cells, lang)
    columns = min(8, max(1, len(cells)))
    sheet = np.full((max(1, (len(cells)+columns-1)//columns)*184, columns*160, 3), 245, np.uint8)
    for i, cell in enumerate(cells):
        crop, _ = crop_cell(image, cell)
        name = f'line_{cell["line_id"]+1:03d}_cell_{cell["line_cell_index"]+1:03d}.png'
        if not cv2.imwrite(str(destination/name), crop):
            raise OSError(f'Cannot save {destination/name}')
        records.append(dict(cell, crop_file=name, token=tokens[i],
                            text_trace=text_trace(tokens[i]['char'])))
        x, y = (i % columns)*160, (i//columns)*184
        sheet[y:y+144, x:x+144] = cv2.resize(crop, (144, 144), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, f'L{cell["line_id"]+1} C{cell["line_cell_index"]+1}',
                    (x+4, y+156), cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1)
        cv2.putText(sheet, 'dots: '+''.join(map(str, sorted(cell['dots']))),
                    (x+4, y+174), cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1)
    if not cv2.imwrite(str(destination/'contact_sheet.png'), sheet):
        raise OSError('Cannot save crop contact sheet')
    lines = []
    for line_id in dict.fromkeys(cell['line_id'] for cell in cells):
        line_cells = [cell for cell in cells if cell['line_id'] == line_id]
        lines.append(dict(line_id=line_id, text=decode_cells(line_cells, lang)))
    report = dict(source=str(source) if source else None, language=lang,
                  image_shape=image.shape, text=decode_cells(cells, lang), lines=lines,
                  input_file='inference_input.png',
                  camera_roi_file='camera_roi.png' if captured_image is not None else None,
                  input_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                  text_trace=text_trace(decode_cells(cells, lang)),
                  warning_counts=dict(Counter(token['warning'] for token in tokens
                                              if token.get('warning') and not token.get('consumed'))),
                  metadata=metadata or {},
                  diagnostics={key: value for key, value in debug_info.items() if key not in ('dots', 'mask')},
                  cells=records)
    if detector is not None:
        font = detector._get_font(size=18, bold=True)
        model_path = Path(detector.model_path)
        training_args = (getattr(detector.model, 'ckpt', None) or {}).get('train_args', {})
        report['model'] = dict(path=str(model_path), sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            overview_imgsz=detector.yolo_imgsz, crop_imgsz=320,
            read_confidence=detector.confidence, proposal_confidence=detector.proposal_confidence,
            tile_size=detector.tile_size,
            training_args={key: training_args.get(key) for key in
                           ('data', 'model', 'imgsz', 'epochs', 'name', 'project')})
        report['rendering'] = dict(engine='Pillow', font_path=str(getattr(font, 'path', 'Pillow default')))
    path = destination/'manifest.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_value), encoding='utf-8')
    return path
