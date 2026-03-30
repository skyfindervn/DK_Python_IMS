"""
debug_pipeline.py — Test toàn bộ OCR pipeline trên maket mẫu.

Quy trình:
  1. Crop maket bằng crop_maket_by_border()
  2. Zone split
  3. OCR từng zone
  4. In kết quả + lưu debug images
"""
import sys, os, io
# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import importlib.util

# Trực tiếp import modules bằng spec loader — tránh __init__.py import router/fastapi
def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

MODULE_DIR = os.path.join(os.path.dirname(__file__), "..", "web_api", "routers", "kiem_tra_in")

zone_splitter = _import_module("zone_splitter", os.path.join(MODULE_DIR, "zone_splitter.py"))
ocr_engine = _import_module("ocr_engine", os.path.join(MODULE_DIR, "ocr_engine.py"))

import cv2
import numpy as np
from pathlib import Path


def crop_maket_by_border_standalone(image):
    """Simplified V9 crop for standalone testing."""
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
    markets_dir = Path(r"d:\onedriver\OneDrive\Bao_bi_carton\Github\DK2IMS\public\upload\markets")
    debug_dir = Path(__file__).parent
    log_file = debug_dir / "debug_output.txt"

    with open(log_file, "w", encoding="utf-8") as log:
        def p(msg=""):
            log.write(msg + "\n")
            log.flush()

        for f in sorted(markets_dir.glob("*.jpg")):
            p(f"\n{'='*70}")
            p(f"  File: {f.name}")
            p(f"{'='*70}")

            img = cv2.imread(str(f))
            p(f"  Original: {img.shape[1]}x{img.shape[0]}")

            # 1. Crop
            cropped = crop_maket_by_border_standalone(img)
            h, w = cropped.shape[:2]
            p(f"  Cropped:  {w}x{h}")

            # 2. Zone split
            zones = zone_splitter.split_into_zones(cropped)
            p(f"  Zones:    {len(zones)}")

            # Draw zone debug
            zone_vis = zone_splitter.draw_zones_debug(cropped, zones)
            debug_path = str(debug_dir / f"debug_{f.stem}_zones.jpg")
            cv2.imwrite(debug_path, zone_vis)
            p(f"  Zone debug saved: {debug_path}")

            # 3. OCR từng zone
            for z in zones:
                p(f"\n  --- Zone {z['label']} ({z['bbox']}) ---")
                blocks = ocr_engine.ocr_zone(z["image"])
                text = ocr_engine.blocks_to_text(blocks)
                if text:
                    for line in text.split('\n'):
                        p(f"    [{z['label']}] {line}")
                else:
                    p(f"    [{z['label']}] (no text detected)")

        p(f"\n{'='*70}")
        p("Done!")

    # Print summary to stdout
    print(f"Log written to: {log_file}")


if __name__ == "__main__":
    main()

