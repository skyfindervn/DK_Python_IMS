# YOLO Trainer — Carton Print Detection

Package huấn luyện **YOLOv8 Segmentation** để phát hiện và crop chính xác vùng in trên carton/màng nhựa, kể cả ảnh chụp nghiêng.

---

## Cấu trúc thư mục

```
apps/yolo_trainer/
  ├── pyproject.toml          ← Khai báo package uv
  ├── data.yaml               ← Config dataset YOLO
  ├── train.py                ← Script huấn luyện
  ├── predict.py              ← Script test & inference
  ├── README.md               ← File này
  │
  ├── datasets/
  │   └── carton_seg/
  │       ├── images/
  │       │   ├── train/      ← Ảnh huấn luyện  (*.jpg, *.png)
  │       │   └── val/        ← Ảnh validation  (*.jpg, *.png)
  │       └── labels/
  │           ├── train/      ← Nhãn polygon YOLO-seg (*.txt)
  │           └── val/
  │
  └── weights/
      └── carton_seg_best.pt  ← Weights tốt nhất sau training
```

---

## Bước 1 — Cài dependencies

```bash
# từ root monorepo
uv add ultralytics --package yolo-trainer
```

Hoặc cài môi trường đơn giản:
```bash
cd apps/yolo_trainer
pip install ultralytics opencv-python-headless
```

---

## Bước 2 — Thu thập và gán nhãn dữ liệu

### Số lượng ảnh tối thiểu
- **200 ảnh** cho kết quả ổn định (đa dạng: nhiều loại sản phẩm, nhiều góc chụp, nhiều điều kiện sáng)
- **500+ ảnh** cho kết quả production-grade

### Cách gán nhãn (Dùng Roboflow — miễn phí)

1. Vào [app.roboflow.com](https://app.roboflow.com) → Tạo project loại **Instance Segmentation**
2. Upload ảnh → Dùng tool **Polygon** để vẽ viền vùng in (4 góc)
3. Gán class `carton_print`
4. Export → Format: **YOLOv8 Segmentation**
5. Copy vào `datasets/carton_seg/`

### Format nhãn (`.txt`)
Mỗi file nhãn tương ứng với 1 ảnh, mỗi dòng là 1 object:
```
0  x1 y1 x2 y2 x3 y3 x4 y4
```
*(Toạ độ normalized 0–1, class 0 = carton_print)*

---

## Bước 3 — Huấn luyện

```bash
cd apps/yolo_trainer

# Bắt đầu (dùng YOLOv8n-seg, nhanh nhất)
python train.py

# Nếu có GPU mạnh hơn (dùng Medium hoặc Large)
python train.py --model yolov8m-seg.pt --epochs 200 --batch 16

# Chỉ định GPU
python train.py --device 0
```

**Models theo thứ tự nhỏ → lớn:**

| Model | kích thước | Khuyến nghị |
|---|---|---|
| `yolov8n-seg.pt` | ~6MB | Bắt đầu, test nhanh |
| `yolov8s-seg.pt` | ~22MB | **Recommended production** |
| `yolov8m-seg.pt` | ~52MB | Khi có GPU ≥ 8GB |
| `yolov8l-seg.pt` | ~87MB | High-end GPU |

---

## Bước 4 — Test inference

```bash
# Test với 1 ảnh
python predict.py --image test.jpg --template maket.jpg

# Chỉ crop (không cần maket)
python predict.py --image test.jpg
```

---

## Bước 5 — Tích hợp vào web_api

Sau khi train xong:
1. Copy `weights/carton_seg_best.pt` → `apps/web_api/models/carton_seg_best.pt`
2. Trong `image_processor.py` của web_api, gọi hàm `detect_and_crop` từ `predict.py`

---

## Tips chụp ảnh khi thu thập

- Chụp từ nhiều góc độ khác nhau (thẳng, nghiêng 15°, 30°, 45°)
- Bao gồm nhiều loại bề mặt: carton nâu, trắng, màng nhựa bóng, giấy phủ
- Bao gồm nhiều điều kiện sáng: ánh đèn nhà máy, ngược sáng nhẹ
- Mỗi sản phẩm chụp ít nhất 10–20 ảnh
