"""Export the exact crop inputs and per-cell decisions for a YOLO stream run."""
import json
from pathlib import Path

import cv2
import numpy as np

from decoder import decode_cells, decode_cells_verbose
from yolo_cell_stream import crop_cell


def _json_value(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def export_stream(image, cells, debug_info, output_dir, lang='thai', source=None):
    if debug_info.get('method') != 'yolo_cell_stream':
        raise ValueError('Crop export requires --mode yolo --yolo-pipeline stream')
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    tokens = decode_cells_verbose(cells, lang)
    columns = min(8, max(1, len(cells)))
    sheet = np.full((max(1, (len(cells)+columns-1)//columns)*184, columns*160, 3), 245, np.uint8)
    for i, cell in enumerate(cells):
        crop, _ = crop_cell(image, cell)
        name = f'line_{cell["line_id"]+1:03d}_cell_{cell["line_cell_index"]+1:03d}.png'
        if not cv2.imwrite(str(destination/name), crop):
            raise OSError(f'Cannot save {destination/name}')
        records.append(dict(cell, crop_file=name, token=tokens[i]))
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
                  diagnostics={key: value for key, value in debug_info.items() if key not in ('dots', 'mask')},
                  cells=records)
    path = destination/'manifest.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_value), encoding='utf-8')
    return path
