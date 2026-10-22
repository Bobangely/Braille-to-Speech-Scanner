"""Fine-tune local YOLO weights on an audited dataset without replacing production weights."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime, timezone
import hashlib
import json

from tools.training.validate_dataset import validate_dataset

ROOT = Path(__file__).resolve().parents[2]


def train(data, weights=None, epochs=30, batch=8, imgsz=640, device='cpu',
          name=None, resume=None, workers=0):
    data = Path(data).resolve()
    if data.is_dir():
        data = data/'data.yaml'
    if data.name != 'data.yaml':
        raise ValueError('Training requires the audited data.yaml, not an alternate YAML file')
    if min(epochs, batch, imgsz) < 1 or imgsz % 32 or workers < 0:
        raise ValueError('Positive epochs/batch, imgsz multiple of 32, and nonnegative workers required')
    report = validate_dataset(data.parent)
    name = name or 'synthetic_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run name must be a directory name, not a path')
    project = ROOT/'runs/detect'
    if not resume and (project/name).exists():
        raise FileExistsError(f'Run already exists: {project/name}')
    checkpoint = Path(resume or weights or ROOT/'models/braille_yolo.pt').resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Local weights missing: {checkpoint}')
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    if model.task != 'detect' or model.names != {0: 'braille_dot'}:
        raise ValueError('Expected a local braille_dot detection checkpoint')
    if resume:
        if weights is not None:
            raise ValueError('Choose --weights for a new round, or --resume for an interrupted round')
        previous_data = Path(model.ckpt.get('train_args', {}).get('data', '')).resolve()
        if previous_data != data:
            raise ValueError('Resume requires the same dataset; use --weights to fine-tune on a new version')
        if model.ckpt.get('epoch', -1) < 0:
            raise ValueError('Checkpoint is from a finished run; start a new round with --weights')
        model.train(resume=True, device=device)
    else:
        model.train(data=str(data), epochs=epochs, batch=batch, imgsz=imgsz, device=device,
                    workers=workers, seed=42, deterministic=True, patience=10,
                    project=str(project), name=name, exist_ok=False, save=True, plots=True,
                    # Geometry augmentation is already applied to images AND metadata.
                    fliplr=0, flipud=0, degrees=0, perspective=0, translate=0, scale=0,
                    mosaic=0, mixup=0, copy_paste=0, close_mosaic=0,
                    hsv_h=0, hsv_s=0, hsv_v=0, amp=False)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir/'weights/best.pt'
    last = save_dir/'weights/last.pt'
    if not best.is_file() or not last.is_file():
        raise RuntimeError('Training did not produce both best.pt and last.pt')
    training_report = dict(dataset=str(data), dataset_audit=report, source_weights=str(checkpoint),
                           source_sha256=before_hash, best_weights=str(best), last_weights=str(last),
                           best_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
                           production_weights_replaced=False, resumed=bool(resume))
    if not resume and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != before_hash:
        raise RuntimeError('Source weights changed during training')
    (save_dir/'training_report.json').write_text(json.dumps(training_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(training_report, ensure_ascii=False, indent=2))
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Dataset directory or data.yaml')
    parser.add_argument('--weights', default=None, help='Previous best.pt for a new training round')
    parser.add_argument('--resume', default=None, help='Interrupted last.pt; requires the same dataset')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu', help='cpu or GPU index such as 0')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    train(**vars(args))
