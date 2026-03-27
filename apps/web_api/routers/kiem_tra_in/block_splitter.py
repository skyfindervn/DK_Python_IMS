"""
block_splitter.py – Logic chia block ảnh phôi theo loại sản phẩm.

Cấu trúc phôi (từ hệ thống DK2IMS):
  Loai 1 (Thùng thường):
    phoi_dai  = (D + R) x 2 + 35mm  (1 tai)
    phoi_rong = D + R
    Lay-out:  Tai(35) | D | R | D | R   → 5 blocks
  Loai 2 (Thùng 2 tấm / 2 tai):
    phoi_dai  = D + R + 40mm  (1 tai 40mm)
    phoi_rong = R + C
    Lay-out:  Tai(40) | D | R          → 3 blocks
  Loai 3 (Tấm bé):
    1 block toàn tấm

Trục X = chiều DÀI phôi, trục Y = chiều RỘNG phôi.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Block:
    no: int
    label: str
    x_ratio: float   # offset x / phoi_dai
    y_ratio: float   # offset y / phoi_rong
    w_ratio: float   # chiều rộng block / phoi_dai
    h_ratio: float   # chiều cao block / phoi_rong


# ── Loại 1: Thùng thường — 5 blocks ──────────────────────────────────────────
#
#  ┌──────┬───────────┬─────────┬───────────┬─────────┐
#  │ Tai  │    D      │    R    │    D      │    R    │
#  │ 35mm │ Mặt c.1  │  Hông1  │ Mặt c.2  │  Hông2  │
#  └──────┴───────────┴─────────┴───────────┴─────────┘
#   Block1   Block2     Block3    Block4     Block5
#
#  phoi_dai = 35 + D + R + D + R = 35 + 2(D+R)

def get_blocks_thuong(phoi_dai: float, phoi_rong: float,
                      D: float, R: float) -> list[Block]:
    """Thùng thường: 5 blocks.  phoi_dai = (D+R)*2 + 35"""
    tai_mm = 35.0

    # Tỉ lệ
    w_tai = tai_mm / phoi_dai
    w_D   = D / phoi_dai
    w_R   = R / phoi_dai

    x0_tai = 0.0
    x1_D   = w_tai
    x2_R   = w_tai + w_D
    x3_D   = w_tai + w_D + w_R
    x4_R   = w_tai + 2 * w_D + w_R

    return [
        Block(no=1, label=f"Tai (35mm)",      x_ratio=x0_tai, y_ratio=0.0, w_ratio=w_tai, h_ratio=1.0),
        Block(no=2, label=f"Mặt chính 1 (D={D:.0f}mm)", x_ratio=x1_D, y_ratio=0.0, w_ratio=w_D,   h_ratio=1.0),
        Block(no=3, label=f"Hông 1 (R={R:.0f}mm)",       x_ratio=x2_R, y_ratio=0.0, w_ratio=w_R,   h_ratio=1.0),
        Block(no=4, label=f"Mặt chính 2 (D={D:.0f}mm)", x_ratio=x3_D, y_ratio=0.0, w_ratio=w_D,   h_ratio=1.0),
        Block(no=5, label=f"Hông 2 (R={R:.0f}mm)",       x_ratio=x4_R, y_ratio=0.0, w_ratio=w_R,   h_ratio=1.0),
    ]


# ── Loại 2: Thùng 2 tấm / 2 tai — 3 blocks ──────────────────────────────────
#
#  ┌──────┬───────────┬─────────┐
#  │ Tai  │    D      │    R    │
#  │ 35mm │ Chiều dài │ Chiều   │
#  │      │  thùng   │  rộng   │
#  └──────┴───────────┴─────────┘
#   Block1   Block2     Block3
#
#  phoi_dai = 35 + D + R
#  (tai sau 40mm theo công thức resolveKichThuocPhoi nhưng không phân tích riêng)

def get_blocks_2tam(phoi_dai: float, phoi_rong: float,
                    D: float, R: float) -> list[Block]:
    """Thùng 2 tấm: 3 blocks.  phoi_dai = D + R + 40"""
    tai_mm = 35.0   # Tai phân tích = 35mm (phần trước)

    w_tai = tai_mm / phoi_dai
    w_D   = D / phoi_dai
    w_R   = R / phoi_dai

    x0 = 0.0
    x1 = w_tai
    x2 = w_tai + w_D

    return [
        Block(no=1, label=f"Tai (35mm)",              x_ratio=x0, y_ratio=0.0, w_ratio=w_tai, h_ratio=1.0),
        Block(no=2, label=f"Chiều dài thùng (D={D:.0f}mm)", x_ratio=x1, y_ratio=0.0, w_ratio=w_D,   h_ratio=1.0),
        Block(no=3, label=f"Chiều rộng thùng (R={R:.0f}mm)",x_ratio=x2, y_ratio=0.0, w_ratio=w_R,   h_ratio=1.0),
    ]


# ── Loại 3: Tấm bé — 1 block ─────────────────────────────────────────────────

def get_blocks_tam_be(phoi_dai: float, phoi_rong: float,
                      D: float, R: float) -> list[Block]:
    """Tấm bé: 1 block toàn tấm."""
    return [
        Block(no=1, label="Toàn tấm", x_ratio=0.0, y_ratio=0.0, w_ratio=1.0, h_ratio=1.0),
    ]


# ── Factory ───────────────────────────────────────────────────────────────────

def get_blocks(loai: int, phoi_dai: float, phoi_rong: float,
               D: float = 0.0, R: float = 0.0) -> list[Block]:
    """
    Factory: lấy danh sách block theo loại sản phẩm.
    D = chieu_dai thùng (mm), R = chieu_rong thùng (mm).
    """
    if loai == 1:
        return get_blocks_thuong(phoi_dai, phoi_rong, D, R)
    elif loai == 2:
        return get_blocks_2tam(phoi_dai, phoi_rong, D, R)
    else:
        return get_blocks_tam_be(phoi_dai, phoi_rong, D, R)


# ── Crop blocks từ ảnh ───────────────────────────────────────────────────────

def crop_blocks(img: np.ndarray, blocks: list[Block]) -> list[tuple[Block, np.ndarray]]:
    """
    Cắt các block từ ảnh đã resize.
    img: numpy (H, W, C) — H=phoi_rong chiều, W=phoi_dai chiều.
    """
    h, w = img.shape[:2]
    results = []
    for blk in blocks:
        x  = int(blk.x_ratio * w)
        y  = int(blk.y_ratio * h)
        bw = int(blk.w_ratio * w)
        bh = int(blk.h_ratio * h)
        x  = max(0, min(x,  w - 1))
        y  = max(0, min(y,  h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        cropped = img[y:y + bh, x:x + bw]
        results.append((blk, cropped))
    return results
