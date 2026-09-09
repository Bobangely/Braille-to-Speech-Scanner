"""Read Thai/English Braille with the YOLO cell stream."""
import argparse
from pathlib import Path
import sys

import cv2

from decoder import decode_cells, decode_cells_verbose
from yolo_detector import YOLOBrailleDetector


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?')
    parser.add_argument('--camera', type=int, nargs='?', const=0)
    parser.add_argument('--model')
    parser.add_argument('--lang', choices=['thai', 'english'], default='thai')
    parser.add_argument('--conf', type=float, default=.35)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--speak', action='store_true')
    args = parser.parse_args()
    if args.camera is not None or args.image is None:
        from camera_reader import RealTimeBrailleScanner
        RealTimeBrailleScanner(camera_id=args.camera or 0, lang=args.lang,
                              yolo_conf=args.conf, model_path=args.model).run()
        return
    image = cv2.imread(args.image)
    if image is None:
        parser.error(f'Cannot read image: {args.image}')
    detector = YOLOBrailleDetector(model_path=args.model, confidence=args.conf)
    cells, debug = detector.detect(image, lang=args.lang)
    text = decode_cells(cells, args.lang)
    print(text or 'ไม่พบข้อความเบรลล์')
    annotated = detector.annotate_with_text(image, debug['dots'], cells, text,
                                            decode_cells_verbose(cells, args.lang), args.lang)
    if args.save:
        destination = Path(__file__).resolve().parent/'output'/(Path(args.image).stem+'_yolo.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), annotated):
            raise OSError(f'Cannot save {destination}')
        print(destination)
    if args.speak and text:
        from tts import speak
        speak(text, lang=args.lang)
    if not args.no_display:
        cv2.namedWindow('Braille YOLO', cv2.WINDOW_NORMAL)
        cv2.imshow('Braille YOLO', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
