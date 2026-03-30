"""
ocr_engine.py — PaddleOCR wrapper cho Vietnamese text detection + recognition.

Singleton PaddleOCR instance, lazy-loaded.
Hỗ trợ đọc text từ ảnh zone, trả về list TextBlock(text, bbox, confidence).
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Singleton OCR instance
_ocr_instance = None


@dataclass
class TextBlock:
    """Thông tin một dòng text OCR detect được."""
    text: str
    bbox: list        # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    confidence: float
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self):
        if self.bbox:
            xs = [p[0] for p in self.bbox]
            ys = [p[1] for p in self.bbox]
            self.center_x = sum(xs) / len(xs)
            self.center_y = sum(ys) / len(ys)


def _get_ocr():
    """Lazy-load PaddleOCR singleton."""
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_instance = PaddleOCR(
                lang="vi",              # Vietnamese
                use_angle_cls=False,    # Tắt angle detector để tránh memory leak/crash trên C++ DLL
            )
            logger.info("PaddleOCR initialized (lang=vi)")
        except Exception as e:
            logger.error(f"Failed to init PaddleOCR: {e}")
            raise
    return _ocr_instance


def ocr_image(image: np.ndarray, min_confidence: float = 0.5) -> List[TextBlock]:
    """
    Chạy OCR trên ảnh, trả về list TextBlock đã lọc theo confidence.

    Args:
        image: ảnh BGR (OpenCV format)
        min_confidence: ngưỡng tối thiểu (0.0-1.0)

    Returns:
        List[TextBlock] sorted theo vị trí (top-to-bottom, left-to-right)
    """
    ocr = _get_ocr()

    # PaddleOCR v3.4 API: predict() thay vì ocr()
    try:
        results = ocr.predict(image)
    except Exception as e:
        logger.warning(f"OCR predict failed: {e}")
        return []

    blocks = []
    if not results:
        return blocks

    # PaddleOCR v3.4 trả về list of dict hoặc generator
    for result in results:
        # Mỗi result có thể là dict với key 'rec_texts', 'rec_scores', 'dt_polys'
        if isinstance(result, dict):
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            polys = result.get("dt_polys", [])

            for i, (text, score) in enumerate(zip(texts, scores)):
                conf = float(score)
                if conf < min_confidence:
                    continue
                if not text.strip():
                    continue

                bbox = polys[i].tolist() if i < len(polys) else [[0,0],[0,0],[0,0],[0,0]]
                blocks.append(TextBlock(
                    text=text.strip(),
                    bbox=bbox,
                    confidence=conf,
                ))
        else:
            # Fallback: legacy format [[bbox, (text, conf)], ...]
            try:
                if hasattr(result, '__iter__'):
                    for line in result:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            bbox = line[0]
                            text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                            conf = float(line[1][1]) if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.5
                            if conf >= min_confidence and text.strip():
                                blocks.append(TextBlock(
                                    text=text.strip(),
                                    bbox=bbox if isinstance(bbox, list) else [[0,0],[0,0],[0,0],[0,0]],
                                    confidence=conf,
                                ))
            except Exception as e:
                logger.warning(f"OCR result parsing fallback failed: {e}")

    # Sort: top-to-bottom, then left-to-right
    blocks.sort(key=lambda b: (b.center_y, b.center_x))

    # Xóa tham chiếu tới C++ objects để giải phóng RAM/GPU nhanh hơn
    del results
    import gc
    gc.collect()

    return blocks


def ocr_zone(zone_image: np.ndarray, preprocess: bool = True) -> List[TextBlock]:
    """
    OCR một zone cụ thể với preprocessing tùy chọn.

    Preprocessing bao gồm:
    - Resize nếu quá lớn (max 2000px cạnh dài) — tránh PaddleOCR crash
    - Resize nếu quá nhỏ (<200px cạnh ngắn) — OCR cần tối thiểu ~50px font
    - Sharpen nhẹ
    """
    if zone_image is None or zone_image.size == 0:
        return []

    h, w = zone_image.shape[:2]

    if preprocess:
        # Resize xuống nếu zone quá lớn — PaddleOCR crash (OOM CPU) trên ảnh quá lớn
        MAX_DIM = 960
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            zone_image = cv2.resize(zone_image, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
            h, w = zone_image.shape[:2]

        # Resize nếu zone quá nhỏ (<200px cạnh ngắn) — OCR cần tối thiểu ~50px font
        if min(h, w) < 200:
            scale = 200 / min(h, w)
            zone_image = cv2.resize(zone_image, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_CUBIC)

        # Sharpen nhẹ
        kernel = np.array([[-0.5, -0.5, -0.5],
                           [-0.5,  5.0, -0.5],
                           [-0.5, -0.5, -0.5]])
        zone_image = cv2.filter2D(zone_image, -1, kernel)

    return ocr_image(zone_image)


def blocks_to_text(blocks: List[TextBlock]) -> str:
    """Ghép tất cả TextBlock thành một chuỗi text, mỗi block một dòng."""
    return "\n".join(b.text for b in blocks)


def draw_ocr_debug(image: np.ndarray, blocks: List[TextBlock]) -> np.ndarray:
    """Vẽ bounding boxes và text lên ảnh debug."""
    vis = image.copy()
    for b in blocks:
        pts = np.array(b.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        # Text label
        x, y = int(b.center_x), int(b.center_y)
        label = f"{b.text[:20]}... ({b.confidence:.2f})" if len(b.text) > 20 else f"{b.text} ({b.confidence:.2f})"
        cv2.putText(vis, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return vis
