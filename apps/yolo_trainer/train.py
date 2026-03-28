"""
train.py — YOLOv8 Segmentation training script
================================================================
Huấn luyện mô hình YOLOv8-seg để phân vùng (segment) object
trên ảnh, ví dụ: detect vùng box/carton với mask polygon.

Sau khi train xong, file weights/best.pt sẽ được dùng
bởi web_api để inference.

Chạy:
    python train.py
    python train.py --model yolov8s-seg.pt --epochs 200 --batch 4
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "yolov8m-seg.pt"        # Segmentation: phân vùng object bằng polygon mask
DEFAULT_EPOCHS = 150
DEFAULT_IMGSZ  = 640
DEFAULT_BATCH  = 8                       # giảm xuống 4 nếu RAM GPU < 8GB
DEFAULT_DATA   = "datasets/data.yaml"    # data.yaml từ Roboflow export
DEFAULT_OUTPUT = "runs/findcarton_seg"
DEFAULT_DEVICE = ""                      # "" = auto detect (GPU > CPU)

# ──────────────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    print(f"\n{'='*60}")
    print(f"  FINDCARTON — YOLOv8 SEGMENTATION TRAINER")
    print(f"{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Img size : {args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Data     : {args.data}")
    print(f"{'='*60}\n")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {data_path}. "
            "Hãy đảm bảo đã gán nhãn và đặt dữ liệu đúng cấu trúc."
        )

    # ── Tự động patch data.yaml với đường dẫn tuyệt đối đúng theo OS ──────────
    import yaml
    dataset_dir = data_path.parent.resolve()
    with open(data_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    yaml_data["path"] = str(dataset_dir)
    with open(data_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False)
    print(f"  Dataset path → {dataset_dir}")
    # ──────────────────────────────────────────────────────────────────────────

    model = YOLO(args.model)

    results = model.train(
        data        = str(data_path),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        project     = args.output,
        name        = "train",
        device      = args.device or None,
        # ── Augmentation: giúp mô hình học ảnh nghiêng, sáng tối ──
        degrees     = 15.0,      # xoay ngẫu nhiên ±15°
        perspective = 0.001,     # biến dạng phối cảnh nhẹ
        flipud      = 0.2,       # flip dọc 20%
        fliplr      = 0.5,       # flip ngang 50%
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        mosaic      = 0.5,       # ghép 4 ảnh thành 1, tăng đa dạng
        close_mosaic= 10,        # tắt mosaic 10 epoch cuối để ổn định
        # ── Optimizer ──
        optimizer   = "AdamW",
        lr0         = 0.001,
        warmup_epochs = 5,
        save_period = 20,        # lưu checkpoint mỗi 20 epoch
        patience    = 30,        # early stopping nếu 30 epoch không cải thiện
        verbose     = True,
    )

    best_weights = Path(args.output) / "train" / "weights" / "best.pt"
    dest = Path("weights") / "findcarton_seg_best.pt"
    dest.parent.mkdir(exist_ok=True)
    if best_weights.exists():
        shutil.copy(best_weights, dest)
        print(f"\n✅ Đã copy weights tốt nhất về: {dest}")
    
    print("\n📊 Validation tự động sau training:")
    model.val()

    print(f"\n🎉 Training hoàn tất! Weights: {dest}\n")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Trainer")
    parser.add_argument("--model",   default=DEFAULT_MODEL,  help="Pretrained model path")
    parser.add_argument("--epochs",  default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--imgsz",   default=DEFAULT_IMGSZ,  type=int)
    parser.add_argument("--batch",   default=DEFAULT_BATCH,  type=int)
    parser.add_argument("--data",    default=DEFAULT_DATA)
    parser.add_argument("--output",  default=DEFAULT_OUTPUT)
    parser.add_argument("--device",  default=DEFAULT_DEVICE, help="'cpu', '0', '0,1'")
    main(parser.parse_args())
