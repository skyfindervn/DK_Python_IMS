"""
router.py – FastAPI router cho API /api/kiem-tra-in

Pipeline OCR-based:
  1. Load ảnh chụp + maket
  2. Crop maket (morphological border detection)
  3. YOLO Seg crop + Perspective warp + Orientation detect
  4. Zone split (chia vùng theo đường gấp)
  5. OCR từng zone (PaddleOCR Vietnamese)
  6. So sánh text maket vs chụp → phát hiện lỗi
"""

import logging
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException

from .schemas import (
    KiemTraInRequest, KiemTraInResponse,
    AnalysisResult, ZoneResult, TextDiffItem,
)
from .image_processor import (
    load_image, image_to_base64, universal_align,
    _push_debug_image, yolo_crop, crop_maket_by_border,
)
from .zone_splitter import split_into_zones, draw_zones_debug
from .ocr_engine import ocr_zone, blocks_to_text
from .text_comparator import compare_zone_texts, compare_all_zones
import log_broker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["kiem-tra-in"])


@router.post("/kiem-tra-in", response_model=KiemTraInResponse)
async def kiem_tra_in(req: KiemTraInRequest) -> KiemTraInResponse:
    """
    Nhận ảnh chụp + maket từ Laravel,
    xử lý bằng OpenCV (crop/align), OCR text comparison, trả kết quả.
    """
    t_start = time.time()

    log_broker.push(
        f"📨 Nhận yêu cầu mới — SP: {req.ma_san_pham or '?'} | Loại: {req.loai} | Phôi: {req.phoi_dai}×{req.phoi_rong}mm",
        level="STEP",
    )

    # ── 1. Load ảnh ────────────────────────────────────────────────────────────
    log_broker.push("📷 [1/6] Đang load ảnh chụp...", level="INFO")
    img_chup = load_image(req.chup_image)
    if img_chup is None:
        log_broker.push(f"❌ Không đọc được ảnh: {req.chup_image}", level="ERR")
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh chụp: {req.chup_image}")

    h0, w0 = img_chup.shape[:2]
    fname = Path(req.chup_image).name
    log_broker.push(
        f"✅ Ảnh nhận OK — {fname} ({w0}×{h0}px)",
        level="OK",
        event_type="image_received",
        path=req.chup_image,
        filename=fname,
    )

    log_broker.push("🖼 [1/6] Đang load ảnh maket...", level="INFO")
    img_maket = load_image(req.maket_image)
    if img_maket is None:
        log_broker.push(f"❌ Không đọc được maket: {req.maket_image}", level="ERR")
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh maket: {req.maket_image}")

    log_broker.push(
        f"🖼 Maket OK — {Path(req.maket_image).name}",
        level="OK",
        event_type="maket_loaded",
        url=f"/api/preview-image?path={req.maket_image}",
        tenmarket=req.ten_maket or "",
    )

    # ── 2. Crop maket: dò viền bằng morphological intersection scoring ────────
    log_broker.push("✂️ [2/6] Đang crop maket (dò viền carton)...", level="INFO")
    maket_crop_res = crop_maket_by_border(img_maket)

    # Fallback sang YOLO nếu không tìm thấy viền
    if maket_crop_res["status"] != "success":
        log_broker.push("⚠️ Không tìm thấy viền → thử YOLO crop maket...", level="WARN")
        maket_crop_res = yolo_crop(img_maket, conf=0.35, model_name="findcarton_maket_best.pt")

    if maket_crop_res["status"] == "success":
        img_maket = maket_crop_res["cropped"]
        h_mk, w_mk = img_maket.shape[:2]
        _push_debug_image(f"✂️ [MAKET] Crop OK ({w_mk}x{h_mk})", img_maket, "yolo_maket")
        log_broker.push(f"✅ Crop maket OK | {w_mk}×{h_mk}px", level="OK")
    else:
        log_broker.push("⚠️ Không crop được maket. Dùng maket gốc.", level="WARN")

    # ── 3. Crop + Align ảnh chụp (YOLO Seg → Perspective warp → Orientation) ──
    log_broker.push("🔧 [3/6] Crop ảnh chụp bằng YOLO → Orientation detect...", level="STEP")

    align_result = universal_align(img_chup, img_maket)
    yolo_conf = align_result.get("yolo_conf", 0.0)
    matches = align_result.get("match_count", 0)

    if yolo_conf > 0:
        log_broker.push(f"✂️ YOLO crop OK | conf={yolo_conf:.2f}", level="OK")

    if align_result["status"] == "success":
        img_chup_aligned = align_result["aligned_image"]
        log_broker.push(f"✅ Alignment OK | {matches} inliers.", level="OK")
    else:
        log_broker.push(f"⚠️ Alignment thất bại ({matches} matches). Dùng ảnh gốc.", level="WARN")
        img_chup_aligned = cv2.resize(img_chup, (img_maket.shape[1], img_maket.shape[0]))

    ha, wa = img_chup_aligned.shape[:2]

    # Resize maket về cùng kích thước
    img_maket_resized = cv2.resize(img_maket, (wa, ha))

    # Push ảnh aligned vào UI
    b64_aligned = image_to_base64(img_chup_aligned)
    log_broker.push(
        f"✅ Ảnh aligned: {wa}×{ha}px | Nền: {req.mau_nen}",
        level="OK",
        event_type="image_processed",
        base64=b64_aligned,
        info=f"Aligned: {wa}×{ha}px | nền={req.mau_nen}",
    )

    # ── 4. Zone Split ──────────────────────────────────────────────────────────
    log_broker.push("📐 [4/6] Chia vùng theo đường gấp carton...", level="STEP")

    zones_maket = split_into_zones(img_maket_resized)
    zones_chup = split_into_zones(img_chup_aligned)

    log_broker.push(
        f"✅ Zones: Maket={len(zones_maket)} | Chụp={len(zones_chup)}",
        level="OK",
    )

    # Debug: vẽ zones lên ảnh
    zone_vis = draw_zones_debug(img_maket_resized, zones_maket)
    _push_debug_image(
        f"📐 [ZONES] Maket split: {len(zones_maket)} zones",
        zone_vis, "zone_split"
    )

    # ── 5. OCR từng zone ───────────────────────────────────────────────────────
    log_broker.push(
        f"🔤 [5/6] OCR {len(zones_maket)} zones (PaddleOCR Vietnamese)...",
        level="STEP",
    )

    maket_texts = {}
    chup_texts = {}

    for z in zones_maket:
        blocks = ocr_zone(z["image"])
        text = blocks_to_text(blocks)
        maket_texts[z["zone_id"]] = text
        if text:
            log_broker.push(
                f"  📖 Maket [{z['label']}]: {text[:80]}{'...' if len(text) > 80 else ''}",
                level="INFO",
            )

    for z in zones_chup:
        blocks = ocr_zone(z["image"])
        text = blocks_to_text(blocks)
        chup_texts[z["zone_id"]] = text

    log_broker.push("✅ OCR hoàn tất", level="OK")

    # ── 6. So sánh text ────────────────────────────────────────────────────────
    log_broker.push("🔍 [6/6] So sánh text giữa maket và bản in...", level="STEP")

    compare_result = compare_all_zones(
        zones_maket, zones_chup,
        maket_texts, chup_texts,
    )

    # Chuyển kết quả thành response schema
    zone_results_response = []
    for zr in compare_result["zone_results"]:
        diffs_response = []
        for d in zr.diffs:
            diffs_response.append(TextDiffItem(
                diff_type=d.diff_type,
                expected=d.expected,
                actual=d.actual,
                similarity=d.similarity,
                detail=d.detail,
            ))

        zone_results_response.append(ZoneResult(
            zone_id=zr.zone_id,
            zone_label=zr.zone_label,
            status=zr.status,
            maket_text=zr.maket_text,
            chup_text=zr.chup_text,
            diffs=diffs_response,
            similarity_score=zr.similarity_score,
            error_summary=zr.error_summary,
        ))

        # Log chi tiết lỗi từng zone
        if zr.status == 0:
            log_broker.push(
                f"  ❌ [{zr.zone_label}] {zr.error_summary}",
                level="WARN",
            )
            for d in zr.diffs:
                if d.diff_type == "missing":
                    log_broker.push(
                        f"    🔴 Thiếu: \"{d.expected}\"",
                        level="WARN",
                    )
                elif d.diff_type == "extra":
                    log_broker.push(
                        f"    🟡 Thừa: \"{d.actual}\"",
                        level="WARN",
                    )
                elif d.diff_type == "modified":
                    log_broker.push(
                        f"    🟠 Sai: \"{d.expected}\" → \"{d.actual}\" ({d.similarity:.0%})",
                        level="WARN",
                    )
        else:
            log_broker.push(
                f"  ✅ [{zr.zone_label}] OK ({zr.similarity_score:.0%})",
                level="OK",
            )

    # Tổng hợp
    overall_status = compare_result["status"]
    n_errors = compare_result["error_zones"]
    n_total = compare_result["total_zones"]
    summary = compare_result["summary"]

    lbl = "✅ ĐẠT" if overall_status == 1 else "❌ PHÁT HIỆN LỖI"
    elapsed = round(time.time() - t_start, 2)
    ket_luan = "dat" if overall_status == 1 else "khong_dat"

    log_broker.push(
        f"    {lbl}: {n_errors}/{n_total} zones lỗi | {summary}",
        level="OK" if overall_status == 1 else "WARN",
    )
    log_broker.push(
        f"📤 Hoàn thành: {elapsed}s | Kết luận: {ket_luan.upper()}",
        level="OK" if ket_luan == "dat" else "WARN",
        event_type="result_done",
        elapsed=elapsed,
        data={"ma_san_pham": req.ma_san_pham or "", "ket_luan": ket_luan},
    )

    return KiemTraInResponse(
        result=AnalysisResult(
            status=overall_status,
            noi_dung_loi=summary if n_errors > 0 else None,
            vi_tri_loi=", ".join(
                zr.zone_label for zr in compare_result["zone_results"] if zr.status == 0
            ) or None,
            raw=f"OCR comparison: {n_errors}/{n_total} zones failed",
        ),
        zone_results=zone_results_response,
        total_zones=n_total,
        error_zones=n_errors,
        message=f"Hoàn thành: {elapsed}s | {ket_luan.upper()} | {n_errors}/{n_total} zones lỗi",
    )


@router.get("/kiem-tra-in/health")
async def health_check() -> dict:
    return {"status": "ok", "service": "kiem-tra-in", "engine": "paddleocr"}
