"""
schemas.py – Pydantic models cho API /api/kiem-tra-in
"""

from pydantic import BaseModel
from typing import Optional, List


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
    # ── LLM (giữ lại để backward compat, không còn bắt buộc) ──────────────
    llm_endpoint: str = ""
    llm_key: str = ""
    llm_prompt: str = ""
    # ── Dashboard meta ────────────────────────────────────────────────────────
    ma_san_pham: Optional[str] = None
    ten_maket: Optional[str] = None


class TextDiffItem(BaseModel):
    """Chi tiết 1 lỗi text trong zone."""
    diff_type: str              # "missing" | "extra" | "modified"
    expected: str = ""          # text trên maket
    actual: str = ""            # text trên ảnh chụp
    similarity: float = 0.0
    detail: str = ""


class ZoneResult(BaseModel):
    """Kết quả so sánh 1 zone."""
    zone_id: int
    zone_label: str             # "R1C2", "R2C3"...
    status: int                 # 0=có lỗi, 1=OK
    maket_text: str = ""
    chup_text: str = ""
    diffs: List[TextDiffItem] = []
    similarity_score: float = 1.0
    error_summary: str = ""


class AnalysisResult(BaseModel):
    """Kết quả phân tích toàn bộ ảnh."""
    status: int              # 0=có lỗi, 1=ok
    noi_dung_loi: Optional[str] = None
    vi_tri_loi: Optional[str] = None
    raw: Optional[str] = None


class KiemTraInResponse(BaseModel):
    result: Optional[AnalysisResult] = None
    zone_results: List[ZoneResult] = []    # Chi tiết từng zone
    total_zones: int = 0
    error_zones: int = 0
    message: str = "success"
    # Backward compat: trả thêm blocks=[] để controller cũ không bị lỗi
    blocks: list = []

