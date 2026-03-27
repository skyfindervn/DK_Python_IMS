"""
router.py â€“ FastAPI router cho API /api/kiem-tra-in

Pipeline:
  1. Load áº£nh chá»¥p + maket  â†’ push event áº£nh nháº­n
  2. Align áº£nh chá»¥p         â†’ push áº£nh Ä‘Ã£ xá»­ lÃ½ (base64)
  3. Resize maket           â†’ push áº£nh maket
  4. Chia block theo loáº¡i SP
  5. Gá»i LLM tá»«ng block     â†’ push tiáº¿n trÃ¬nh
  6. Tráº£ vá» JSON            â†’ push káº¿t quáº£ cuá»‘i
"""

import logging
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException

from .schemas import KiemTraInRequest, KiemTraInResponse, BlockResult
from .image_processor import load_image, image_to_base64, universal_align
from .block_splitter import get_blocks, crop_blocks
from .llm_client import call_llm
import log_broker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["kiem-tra-in"])


@router.post("/kiem-tra-in", response_model=KiemTraInResponse)
async def kiem_tra_in(req: KiemTraInRequest) -> KiemTraInResponse:
    """
    Nháº­n áº£nh chá»¥p + maket tá»« Laravel,
    xá»­ lÃ½ báº±ng OpenCV + LLM, tráº£ vá» káº¿t quáº£ tá»«ng block.
    """
    t_start = time.time()

    log_broker.push(
        f"ðŸ“¨ Nháº­n yÃªu cáº§u má»›i â€” SP: {req.ma_san_pham or '?'} | Loáº¡i: {req.loai} | PhÃ´i: {req.phoi_dai}Ã—{req.phoi_rong}mm",
        level="STEP",
    )

    # â”€â”€ 1. Load áº£nh â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log_broker.push("ðŸ“· [1/5] Äang load áº£nh chá»¥p...", level="INFO")
    img_chup = load_image(req.chup_image)
    if img_chup is None:
        log_broker.push(f"âŒ KhÃ´ng Ä‘á»c Ä‘Æ°á»£c áº£nh: {req.chup_image}", level="ERR")
        raise HTTPException(status_code=400, detail=f"KhÃ´ng Ä‘á»c Ä‘Æ°á»£c áº£nh chá»¥p: {req.chup_image}")

    h0, w0 = img_chup.shape[:2]
    fname = Path(req.chup_image).name
    log_broker.push(
        f"âœ… áº¢nh nháº­n OK â€” {fname} ({w0}Ã—{h0}px)",
        level="OK",
        event_type="image_received",
        path=req.chup_image,
        filename=fname,
    )

    log_broker.push("ðŸ–¼ [1/5] Äang load áº£nh maket...", level="INFO")
    img_maket = load_image(req.maket_image)
    if img_maket is None:
        log_broker.push(f"âŒ KhÃ´ng Ä‘á»c Ä‘Æ°á»£c maket: {req.maket_image}", level="ERR")
        raise HTTPException(status_code=400, detail=f"KhÃ´ng Ä‘á»c Ä‘Æ°á»£c áº£nh maket: {req.maket_image}")

    log_broker.push(
        f"ðŸ–¼ Maket OK â€” {Path(req.maket_image).name}",
        level="OK",
        event_type="maket_loaded",
        url=f"/api/preview-image?path={req.maket_image}",
        tenmarket=req.ten_maket or "",
    )

    # ── 2. Alignment (Universal SIFT Homography) ───────────────────────────────
    log_broker.push(
        f"🔧 [2/5] Universal Alignment (Template: {img_maket.shape[1]}x{img_maket.shape[0]})...",
        level="STEP",
    )
    
    align_result = universal_align(img_chup, img_maket)
    
    if align_result["status"] == "success":
        img_chup_aligned = align_result["aligned_image"]
        matches = align_result["match_count"]
        log_broker.push(f"✅ Alignment OK | {matches} inliers.", level="OK")
    else:
        # Fallback raw
        log_broker.push(f"⚠️ Alignment thất bại. Dùng ảnh gốc.", level="WARN")
        img_chup_aligned = img_chup
        # Để tránh lỗi tỉ lệ, resize về bằng maket nếu alignment fail
        img_chup_aligned = cv2.resize(img_chup_aligned, (img_maket.shape[1], img_maket.shape[0]))

    ha, wa = img_chup_aligned.shape[:2]
    
    # ── 3. Resize maket chuẩn ───────────────────────────────────────────────
    log_broker.push("🔍 [3/5] Resize maket về kích thước chuẩn...", level="INFO")
    img_maket_resized = cv2.resize(img_maket, (wa, ha))
    log_broker.push(f"✅ Kích thước đồng bộ: {wa}×{ha}px", level="OK")

    b64_aligned = image_to_base64(img_chup_aligned)
    log_broker.push(
        f"✅ Ảnh aligned: {wa}×{ha}px | Nền: {req.mau_nen}",
        level="OK",
        event_type="image_processed",
        base64=b64_aligned,
        info=f"Aligned: {wa}×{ha}px | nền={req.mau_nen}",
    )

    # â”€â”€ 4. Chia block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    blocks = get_blocks(req.loai, req.phoi_dai, req.phoi_rong, D=req.chieu_dai, R=req.chieu_rong)
    log_broker.push(f"ðŸ”² [4/5] Chia {len(blocks)} block theo loáº¡i SP: {req.loai}", level="STEP")

    blocks_chup  = crop_blocks(img_chup_aligned, blocks)
    blocks_maket = crop_blocks(img_maket_resized, blocks)

    # â”€â”€ 5. Gá»i LLM tá»«ng block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log_broker.push(f"ðŸ¤– [5/5] Gá»i LLM cho {len(blocks)} block...", level="STEP")
    results: list[BlockResult] = []

    for idx, ((blk, crop_chup), (_, crop_maket)) in enumerate(zip(blocks_chup, blocks_maket)):
        log_broker.push(
            f"  âŸ¶ Block {blk.no}/{len(blocks)}: {blk.label}",
            level="INFO",
        )
        b64_chup  = image_to_base64(crop_chup)
        b64_maket = image_to_base64(crop_maket)

        llm_result = call_llm(
            endpoint=req.llm_endpoint,
            api_key=req.llm_key,
            prompt_template=req.llm_prompt,
            img_chup_b64=b64_chup,
            img_maket_b64=b64_maket,
            block_label=f"Block {blk.no} â€“ {blk.label}",
        )
        status = llm_result.get("status", 0)
        lbl = "âœ… OK" if status == 1 else "âŒ Lá»–I"
        log_broker.push(
            f"    {lbl} Block {blk.no}: {llm_result.get('noi_dung_loi') or 'â€“'}",
            level="OK" if status == 1 else "WARN",
        )
        results.append(BlockResult(
            no=blk.no,
            status=status,
            noi_dung_loi=llm_result.get("noi_dung_loi"),
            vi_tri_loi=llm_result.get("vi_tri_loi"),
            raw=llm_result.get("raw"),
        ))

    ok_count = sum(1 for r in results if r.status == 1)
    elapsed = round(time.time() - t_start, 2)
    ket_luan = "dat" if ok_count == len(results) else "khong_dat"

    log_broker.push(
        f"ðŸ“¤ HoÃ n thÃ nh: {ok_count}/{len(results)} block OK | {elapsed}s | Káº¿t luáº­n: {ket_luan.upper()}",
        level="OK" if ket_luan == "dat" else "WARN",
        event_type="result_done",
        elapsed=elapsed,
        data={
            "ma_san_pham": req.ma_san_pham or "",
            "ket_luan": ket_luan,
            "blocks": [
                {
                    "block_id": f"Blk {r.no}",
                    "status": "ok" if r.status == 1 else "fail",
                    "error_content": r.noi_dung_loi or "",
                }
                for r in results
            ],
        },
    )

    return KiemTraInResponse(
        blocks=results,
        message=f"HoÃ n thÃ nh: {ok_count}/{len(results)} block OK | {elapsed}s",
    )


@router.get("/kiem-tra-in/health")
async def health_check() -> dict:
    return {"status": "ok", "service": "kiem-tra-in"}

