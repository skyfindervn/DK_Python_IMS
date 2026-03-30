"""
block_analyzer.py – Phân tích sai lệch từng block bằng OpenCV (không cần LLM).

So sánh block ảnh chụp vs block maket:
  - Sai lệch vị trí nội dung (SSIM)
  - Phát hiện vùng khác biệt (diff mask)
  - Đánh giá mức độ lệch màu sắc
  - Tỉ lệ vùng khác biệt (%)

Kết quả được ghép vào prompt gửi LLM để tăng độ chính xác.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BlockAnalysis:
    block_no: int
    diff_ratio: float          # Tỉ lệ % vùng khác biệt (0-100)
    ssim_score: float          # SSIM similarity (0-1, càng cao càng giống)
    color_diff: float          # Chênh lệch màu trung bình (0-255)
    has_significant_diff: bool # True nếu khác biệt đáng kể
    diff_image_b64: str        # Ảnh diff base64 để debug


def analyze_block(
    crop_chup: np.ndarray,
    crop_maket: np.ndarray,
    block_no: int,
    diff_threshold: float = 30.0,  # Ngưỡng pixel khác biệt
    area_threshold: float = 5.0,   # Ngưỡng % vùng khác biệt
) -> BlockAnalysis:
    """
    So sánh block ảnh chụp với block maket bằng OpenCV.
    Trả về BlockAnalysis với các chỉ số đánh giá.
    """
    # Resize về cùng kích thước
    h, w = crop_maket.shape[:2]
    chup_r = cv2.resize(crop_chup, (w, h))

    # ── 1. Grayscale diff ─────────────────────────────────────────────
    gray_maket = cv2.cvtColor(crop_maket, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_chup  = cv2.cvtColor(chup_r, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = np.abs(gray_maket - gray_chup)
    diff_mask = (diff > diff_threshold).astype(np.uint8) * 255

    # Morphology để gộp vùng nhỏ (noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)

    diff_ratio = float(np.sum(diff_mask > 0)) / (w * h) * 100.0

    # ── 2. SSIM-like score ────────────────────────────────────────────
    # Dùng correlation thay vì SSIM đầy đủ (nhanh hơn)
    try:
        corr = np.corrcoef(gray_maket.flatten(), gray_chup.flatten())[0, 1]
        ssim_score = float(max(0.0, corr))
    except Exception:
        ssim_score = 0.0

    # ── 3. Color difference ───────────────────────────────────────────
    chup_lab  = cv2.cvtColor(chup_r, cv2.COLOR_BGR2LAB).astype(np.float32)
    maket_lab = cv2.cvtColor(crop_maket, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_diff = float(np.mean(np.abs(chup_lab - maket_lab)))

    # ── 4. Tạo ảnh diff trực quan ────────────────────────────────────
    diff_vis = chup_r.copy()
    diff_vis[diff_mask > 0] = [0, 0, 255]  # Vùng khác biệt → đỏ

    # Encode base64
    import base64
    _, buf = cv2.imencode(".jpg", diff_vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
    diff_b64 = base64.b64encode(buf).decode()

    has_diff = diff_ratio > area_threshold

    logger.info(
        f"Block {block_no}: diff={diff_ratio:.1f}% | ssim={ssim_score:.3f} | "
        f"color_diff={color_diff:.1f} | significant={has_diff}"
    )

    return BlockAnalysis(
        block_no=block_no,
        diff_ratio=round(diff_ratio, 2),
        ssim_score=round(ssim_score, 4),
        color_diff=round(color_diff, 2),
        has_significant_diff=has_diff,
        diff_image_b64=diff_b64,
    )


def build_analysis_context(analysis: BlockAnalysis) -> str:
    """
    Tạo đoạn text mô tả kết quả phân tích OpenCV để gắn vào prompt LLM.
    Giúp LLM tập trung vào vùng có vấn đề.
    """
    lines = [
        f"[Phân tích kỹ thuật tự động - Block {analysis.block_no}]",
        f"- Tỉ lệ vùng khác biệt: {analysis.diff_ratio:.1f}%"
        + (" ← ĐÁNG KỂ" if analysis.has_significant_diff else " (bình thường)"),
        f"- Độ tương đồng hình ảnh: {analysis.ssim_score:.3f}/1.0"
        + (" ← THẤP" if analysis.ssim_score < 0.7 else ""),
        f"- Chênh lệch màu sắc: {analysis.color_diff:.1f}/255"
        + (" ← LỚN" if analysis.color_diff > 30 else ""),
    ]
    if analysis.has_significant_diff:
        lines.append("→ Phát hiện sai lệch đáng kể. Hãy kiểm tra kỹ nội dung chữ và vị trí.")
    else:
        lines.append("→ Hình ảnh tương đối giống nhau. Kiểm tra chi tiết nhỏ.")
    return "\n".join(lines)


def quick_verdict(analysis: BlockAnalysis) -> dict:
    """
    Đánh giá nhanh chỉ dựa trên OpenCV (không cần LLM).
    Dùng khi LLM không có cấu hình.
    """
    if analysis.diff_ratio > 20.0 or analysis.ssim_score < 0.5:
        status = 0
        noi_dung_loi = (
            f"Phát hiện sai lệch lớn: {analysis.diff_ratio:.1f}% vùng ảnh khác biệt "
            f"(độ tương đồng: {analysis.ssim_score:.2f}). "
            "Cần kiểm tra thủ công."
        )
        vi_tri_loi = "Vùng đỏ trong ảnh diff"
    elif analysis.diff_ratio > 5.0 or analysis.ssim_score < 0.75:
        status = 0
        noi_dung_loi = (
            f"Có sai khác nhỏ: {analysis.diff_ratio:.1f}% vùng ảnh khác biệt. "
            "Có thể do: ánh sáng, màu in nhạt, hoặc thiếu/sai chữ nhỏ."
        )
        vi_tri_loi = None
    else:
        status = 1
        noi_dung_loi = None
        vi_tri_loi = None

    return {
        "status": status,
        "noi_dung_loi": noi_dung_loi,
        "vi_tri_loi": vi_tri_loi,
        "raw": f"OpenCV only: diff={analysis.diff_ratio:.1f}% ssim={analysis.ssim_score:.3f}",
    }
