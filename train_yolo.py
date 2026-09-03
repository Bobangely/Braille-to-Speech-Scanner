"""
YOLO Training Script for Braille Dot Detection
=================================================
สคริปต์สำหรับ train YOLOv8 model ให้ตรวจจับจุดเบรลล์

ขั้นตอนการทำงาน:
    1. เช็คว่ามี training data (datasets/braille_dots/) หรือยัง
       ถ้ายังไม่มี จะสร้างให้อัตโนมัติ
    2. โหลด YOLOv8n (nano) pre-trained model จาก Ultralytics
    3. Train บน dataset ที่สร้าง
    4. บันทึก model ไว้ที่ runs/detect/train/weights/best.pt

การใช้งาน:
    .venv\\Scripts\\python.exe train_yolo.py              ← train ด้วยค่า default
    .venv\\Scripts\\python.exe train_yolo.py --epochs 50  ← กำหนดจำนวน epochs
    .venv\\Scripts\\python.exe train_yolo.py --resume     ← resume training จากครั้งก่อน

YOLO Training Parameters อธิบาย:
    - epochs: จำนวนรอบ training ทั้งหมด (มากขึ้น = แม่นยำขึ้น แต่ช้าลง)
    - batch: จำนวนภาพต่อ batch (ใหญ่ขึ้น = ใช้ RAM/VRAM มากขึ้น แต่เร็วขึ้น)
    - imgsz: ขนาดภาพ input สำหรับ YOLO (640 = มาตรฐาน)
    - device: อุปกรณ์ที่ใช้ train ('0' = GPU, 'cpu' = CPU)
    - patience: หยุด train ถ้า mAP ไม่ดีขึ้นติดต่อกัน N epochs
"""

import os
import sys
import argparse


def check_dataset(data_dir='datasets/braille_dots'):
    """
    ตรวจสอบว่ามี training data หรือยัง
    ถ้ายังไม่มี จะสร้างให้อัตโนมัติ
    """
    yaml_path = os.path.join(data_dir, 'data.yaml')

    if os.path.exists(yaml_path):
        # นับจำนวนภาพ
        train_dir = os.path.join(data_dir, 'images', 'train')
        val_dir = os.path.join(data_dir, 'images', 'val')
        n_train = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
        n_val = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0

        if n_train > 0 and n_val > 0:
            print(f"  ✅ Dataset พร้อมแล้ว!")
            print(f"     Training: {n_train} ภาพ")
            print(f"     Validation: {n_val} ภาพ")
            return yaml_path

    print(f"  ⚠️ ไม่พบ dataset — กำลังสร้างอัตโนมัติ...")
    print()

    from generate_yolo_training import generate_dataset
    generate_dataset(output_dir=data_dir, num_train=800, num_val=200)

    return os.path.join(data_dir, 'data.yaml')


def train(epochs=30, batch=16, imgsz=640, device=None, resume=False):
    """
    Train YOLOv8 model สำหรับตรวจจับจุดเบรลล์

    Parameters
    ----------
    epochs : int
        จำนวนรอบ training (default: 30)
        - 10-20: ทดลองเร็วๆ
        - 30-50: คุณภาพดี
        - 100+: คุณภาพสูงสุด (ใช้เวลานาน)
    batch : int
        จำนวนภาพต่อ batch (default: 16)
        - ถ้า GPU memory ไม่พอ ให้ลดเป็น 8 หรือ 4
    imgsz : int
        ขนาดภาพ input (default: 640)
    device : str
        อุปกรณ์ ('0' = GPU, 'cpu' = CPU, None = auto-detect)
    resume : bool
        ถ้า True จะ resume training จากครั้งก่อน
    """
    from ultralytics import YOLO

    print()
    print("=" * 60)
    print("  🧠 YOLO Braille Dot Detector — Training")
    print("=" * 60)
    print()

    # 1. เช็ค dataset
    data_yaml = check_dataset()
    print()

    # 2. โหลด pre-trained YOLOv8 nano model
    #    YOLOv8n = รุ่นเล็กสุด (3.2M parameters)
    #    เหมาะกับ:
    #    - Dataset ขนาดเล็ก-กลาง
    #    - ต้องการ inference เร็ว (real-time)
    #    - Train ได้บน CPU (ช้าหน่อย) หรือ GPU (เร็วมาก)
    if resume:
        model_path = 'runs/detect/train/weights/last.pt'
        if not os.path.exists(model_path):
            print(f"  ⚠️ ไม่พบ checkpoint สำหรับ resume: {model_path}")
            print(f"     จะเริ่ม training ใหม่...")
            model = YOLO('yolov8n.pt')  # โหลด pre-trained จาก COCO
        else:
            model = YOLO(model_path)
            print(f"  📦 Resuming from: {model_path}")
    else:
        print(f"  📦 โหลด YOLOv8n pre-trained model...")
        model = YOLO('yolov8n.pt')

    # 3. เริ่ม Training
    #    Transfer Learning: เริ่มจาก weights ที่เรียนรู้จาก COCO dataset แล้ว
    #    (80+ classes ของ object ทั่วไป เช่น คน, รถ, แมว)
    #    แล้ว fine-tune ให้ detect "braille_dot" โดยเฉพาะ
    print(f"\n  🏋️ เริ่ม Training...")
    print(f"     Epochs:  {epochs}")
    print(f"     Batch:   {batch}")
    print(f"     ImgSize: {imgsz}")
    print(f"     Device:  {device or 'auto'}")
    print()

    # Train!
    # - data: path ไปยัง data.yaml
    # - epochs: จำนวนรอบ
    # - batch: จำนวนภาพต่อ batch
    # - imgsz: ขนาดภาพ
    # - patience: early stopping (หยุดถ้า mAP ไม่ดีขึ้น N epochs ติดต่อกัน)
    # - save: บันทึก checkpoint ทุก epoch
    # - plots: สร้าง training plots (loss, mAP)
    # - project/name: โฟลเดอร์ output
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device or '',   # '' = auto-detect GPU/CPU
        patience=10,           # หยุดถ้า mAP ไม่ดีขึ้น 10 epochs
        save=True,
        plots=True,
        project='runs/detect',
        name='train',
        exist_ok=True,         # เขียนทับโฟลเดอร์เดิมได้
        verbose=True,
    )

    # 4. สรุปผล
    print()
    print("=" * 60)
    print("  ✅ Training เสร็จสมบูรณ์!")
    print("=" * 60)
    print(f"  📁 Model:      runs/detect/train/weights/best.pt")
    print(f"  📊 Plots:      runs/detect/train/")
    print()
    print(f"  ขั้นตอนถัดไป:")
    print(f"    # ทดสอบ model กับภาพ")
    print(f"    .venv\\Scripts\\python.exe yolo_detector.py sample_images/test_thai_home.png")
    print()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO for Braille Dot Detection')
    parser.add_argument('--epochs', type=int, default=30, help='จำนวนรอบ training (default: 30)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--device', type=str, default=None, help='Device: "0"=GPU, "cpu"=CPU')
    parser.add_argument('--resume', action='store_true', help='Resume training จากครั้งก่อน')
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,
    )
