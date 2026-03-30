"""
debug_full_test.py — Test full OCR pipeline: Maket vs Ảnh chụp thực tế
  - Maket: DTK Plastic 2SP218062-1.jpg
  - Chụp:  A2.JPG (ảnh chụp thực tế trên bàn)

Approach: Zone split MAKET → áp cùng grid lên ảnh chụp → OCR → Compare
"""
import sys, os, io, traceback, time
import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

MODULE_DIR = os.path.join(os.path.dirname(__file__), "..", "web_api", "routers", "kiem_tra_in")
zone_splitter = _import_module("zone_splitter", os.path.join(MODULE_DIR, "zone_splitter.py"))
ocr_engine = _import_module("ocr_engine", os.path.join(MODULE_DIR, "ocr_engine.py"))
text_comparator = _import_module("text_comparator", os.path.join(MODULE_DIR, "text_comparator.py"))

import cv2
import numpy as np
from pathlib import Path


def crop_maket_by_border(image):
    """Simplified V9 crop — tách maket khỏi viền/dimension."""
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    all_pixels = cv2.bitwise_or(binary, cv2.bitwise_or(mask_r1, mask_r2))
    all_pixels = cv2.bitwise_or(all_pixels, mask_blue)

    h_kl = max(w_img // 15, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kl, 1))
    h_morph = cv2.morphologyEx(all_pixels, cv2.MORPH_OPEN, h_kernel)
    v_kl = max(h_img // 15, 100)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kl))
    v_morph = cv2.morphologyEx(all_pixels, cv2.MORPH_OPEN, v_kernel)

    row_density = np.sum(h_morph > 0, axis=1).astype(float)
    col_density = np.sum(v_morph > 0, axis=0).astype(float)
    min_h_span = int(w_img * 0.35)
    min_v_span = int(h_img * 0.25)
    h_cands = np.where(row_density >= min_h_span)[0]
    v_cands = np.where(col_density >= min_v_span)[0]

    def cluster(vals, gap=25):
        if len(vals) == 0: return []
        groups, curr = [], [int(vals[0])]
        for v in vals[1:]:
            v = int(v)
            if v - curr[-1] <= gap: curr.append(v)
            else: groups.append(curr); curr = [v]
        groups.append(curr)
        return groups

    h_groups = cluster(h_cands)
    v_groups = cluster(v_cands)
    h_lines = [(int(np.mean(g)), row_density[int(np.mean(g))]) for g in h_groups]
    v_lines = [(int(np.mean(g)), col_density[int(np.mean(g))]) for g in v_groups]

    edge_h = int(h_img * 0.04)
    edge_w = int(w_img * 0.04)
    h_lines = [l for l in h_lines if edge_h < l[0] < h_img - edge_h]
    v_lines = [l for l in v_lines if edge_w < l[0] < w_img - edge_w]

    if len(h_lines) < 2 or len(v_lines) < 2:
        return image

    intersection_map = cv2.bitwise_and(
        cv2.dilate(h_morph, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))),
        cv2.dilate(v_morph, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    )

    h_int = {}
    for hy, _ in h_lines:
        c = 0
        for vx, _ in v_lines:
            p = intersection_map[max(0,hy-15):min(h_img,hy+16), max(0,vx-15):min(w_img,vx+16)]
            if np.sum(p > 0) > 5: c += 1
        h_int[hy] = c
    v_int = {}
    for vx, _ in v_lines:
        c = 0
        for hy, _ in h_lines:
            p = intersection_map[max(0,hy-15):min(h_img,hy+16), max(0,vx-15):min(w_img,vx+16)]
            if np.sum(p > 0) > 5: c += 1
        v_int[vx] = c

    max_h = max(h_int.values()) if h_int else 0
    max_v = max(v_int.values()) if v_int else 0
    h_th = max(2, max_h * 0.4)
    v_th = max(2, max_v * 0.4)

    h_border = [(y, d) for y, d in h_lines if h_int.get(y, 0) >= h_th]
    v_border = [(x, d) for x, d in v_lines if v_int.get(x, 0) >= v_th]

    if len(h_border) < 2 or len(v_border) < 2:
        h_border = sorted(h_lines, key=lambda l: h_int.get(l[0], 0), reverse=True)
        v_border = sorted(v_lines, key=lambda l: v_int.get(l[0], 0), reverse=True)

    h_pos = sorted([l[0] for l in h_border])
    v_pos = sorted([l[0] for l in v_border])
    yt, yb = h_pos[0], h_pos[-1]
    xl, xr = v_pos[0], v_pos[-1]

    pad = 5
    return image[yt+pad:yb-pad, xl+pad:xr-pad]


def main():
    debug_dir = Path(__file__).parent
    log_file = debug_dir / "debug_full_test.txt"

    MAKET = r"D:\onedriver\OneDrive\Bao_bi_carton\Github\DK2IMS\public\upload\markets\2SP218062-1.jpg"
    CHUP = r"C:\Users\Dung.NT\Pictures\A2.JPG"

    with open(log_file, "w", encoding="utf-8") as log:
        def p(msg=""):
            log.write(msg + "\n")
            log.flush()

        t0 = time.time()

        p("=" * 70)
        p("  FULL OCR PIPELINE TEST: DTK Plastic")
        p("=" * 70)

        # ── 1. Load ảnh ──
        p(f"\n[1] Loading images...")
        img_maket = cv2.imread(MAKET)
        img_chup = cv2.imread(CHUP)
        p(f"  Maket: {img_maket.shape[1]}x{img_maket.shape[0]}")
        p(f"  Chụp:  {img_chup.shape[1]}x{img_chup.shape[0]}")

        # ── 2. Crop maket ──
        p(f"\n[2] Crop maket (remove dimension annotations)...")
        maket_cropped = crop_maket_by_border(img_maket)
        hm, wm = maket_cropped.shape[:2]
        p(f"  Maket cropped: {wm}x{hm}")
        cv2.imwrite(str(debug_dir / "test_maket_cropped.jpg"), maket_cropped)

        # ── 3. Zone Split MAKET ──
        p(f"\n[3] Zone split maket...")
        zones_maket = zone_splitter.split_into_zones(maket_cropped)
        p(f"  Maket zones: {len(zones_maket)}")
        for z in zones_maket:
            p(f"    {z['label']} bbox={z['bbox']}")

        # Save zone debug image
        zone_vis_m = zone_splitter.draw_zones_debug(maket_cropped, zones_maket)
        cv2.imwrite(str(debug_dir / "test_zones_maket.jpg"), zone_vis_m)

        # ── 4. Resize ảnh chụp → cùng kích thước maket cropped ──
        p(f"\n[4] Resize ảnh chụp → {wm}x{hm}...")
        chup_resized = cv2.resize(img_chup, (wm, hm))
        cv2.imwrite(str(debug_dir / "test_chup_resized.jpg"), chup_resized)

        # ── 5. ÁP DỤNG ZONE GRID MAKET lên ảnh chụp ──
        p(f"\n[5] Áp dụng zone grid maket lên ảnh chụp...")
        zones_chup = text_comparator.apply_maket_zones_to_image(chup_resized, zones_maket)
        p(f"  Chụp zones (from maket grid): {len(zones_chup)}")

        # Draw zone overlay on photo
        zone_vis_c = zone_splitter.draw_zones_debug(chup_resized, zones_chup)
        cv2.imwrite(str(debug_dir / "test_zones_chup.jpg"), zone_vis_c)

        # ── 6. OCR ──
        p(f"\n[6] OCR (PaddleOCR Vietnamese)...")
        maket_texts = {}
        chup_texts = {}

        p(f"\n  --- MAKET OCR ---")
        for z in zones_maket:
            try:
                blocks = ocr_engine.ocr_zone(z["image"])
                text = ocr_engine.blocks_to_text(blocks)
            except Exception as e:
                p(f"  [{z['label']}] OCR ERROR: {e}")
                text = ""
            maket_texts[z["zone_id"]] = text
            p(f"  [{z['label']}] {text[:120] if text else '(empty)'}")

        p(f"\n  --- CHỤP OCR ---")
        for z in zones_chup:
            try:
                blocks = ocr_engine.ocr_zone(z["image"])
                text = ocr_engine.blocks_to_text(blocks)
            except Exception as e:
                p(f"  [{z['label']}] OCR ERROR: {e}")
                text = ""
            chup_texts[z["zone_id"]] = text
            p(f"  [{z['label']}] {text[:120] if text else '(empty)'}")

        # ── 7. Text Comparison ──
        p(f"\n[7] Text comparison...")
        result = text_comparator.compare_all_zones(
            zones_maket, zones_chup,
            maket_texts, chup_texts,
        )

        p(f"\n  OVERALL: {'OK' if result['status'] == 1 else 'CÓ LỖI'}")
        p(f"  Total zones (có text): {result['total_zones']}")
        p(f"  Error zones: {result['error_zones']}")
        p(f"  Summary: {result['summary']}")

        p(f"\n  --- ZONE DETAILS ---")
        for zr in result["zone_results"]:
            # Bỏ qua zone không có text
            if not zr.maket_text.strip() and not zr.chup_text.strip():
                continue

            icon = "✅" if zr.status == 1 else "❌"
            p(f"  {icon} [{zr.zone_label}] similarity={zr.similarity_score:.0%}")
            if zr.maket_text:
                p(f"      Maket: {zr.maket_text[:80]}")
            if zr.chup_text:
                p(f"      Chụp:  {zr.chup_text[:80]}")
            if zr.status == 0:
                p(f"      → {zr.error_summary}")
                for d in zr.diffs:
                    if d.diff_type == "missing":
                        p(f"      🔴 Thiếu: \"{d.expected}\"")
                    elif d.diff_type == "extra":
                        p(f"      🟡 Thừa: \"{d.actual}\"")
                    elif d.diff_type == "modified":
                        p(f"      🟠 Sai: \"{d.expected}\" → \"{d.actual}\" ({d.similarity:.0%})")

        elapsed = round(time.time() - t0, 2)
        p(f"\n  ⏱ Total time: {elapsed}s")
        p("=" * 70)
        p("Done!")

    print(f"Full test log written to: {log_file}")


if __name__ == "__main__":
    main()
