"""
schemas.py – Pydantic models cho API /api/kiem-tra-in
"""

from pydantic import BaseModel
from typing import Optional


class KiemTraInRequest(BaseModel):
    chup_image: str          # Đường dẫn tuyệt đối ảnh chụp thực tế
    maket_image: str         # Đường dẫn tuyệt đối ảnh maket
    loai: int                # 1=Thùng thường, 2=Thùng 2 tấm, 3=Tấm bé
    phoi_dai: float          # Kích thước phôi dài (mm)
    phoi_rong: float         # Kích thước phôi rộng (mm)
    # ── Kích thước thùng (D x R x C) ─────────────────────────────────────────
    chieu_dai: float         # D – Chiều dài thùng (mm)
    chieu_rong: float        # R – Chiều rộng thùng (mm)
    chieu_cao: float = 0.0   # C – Chiều cao thùng (mm, optional)
    # ── Màu sắc ───────────────────────────────────────────────────────────────
    mau_nen: str             # "trang" hoặc "xanh"
    mau_giay: str            # "trang" | "nau" | "trang_phu"
    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_endpoint: str
    llm_key: str
    llm_prompt: str
    # ── Dashboard meta ────────────────────────────────────────────────────────
    ma_san_pham: Optional[str] = None
    ten_maket: Optional[str] = None


class BlockResult(BaseModel):
    no: int
    status: int              # 0=lỗi, 1=ok
    noi_dung_loi: Optional[str] = None
    vi_tri_loi: Optional[str] = None
    raw: Optional[str] = None


class KiemTraInResponse(BaseModel):
    blocks: list[BlockResult]
    message: str = "success"
