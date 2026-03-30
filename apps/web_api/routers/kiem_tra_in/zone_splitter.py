"""
zone_splitter.py — Chia ảnh maket/chụp thành các zone theo đường gấp carton.

Sử dụng morphological line detection (tương tự crop_maket_by_border)
để tìm đường gấp H/V bên trong ảnh đã crop, tạo grid cells.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def _detect_lines(image: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect đường ngang và đường dọc bên trong ảnh đã crop.
    Trả về (h_positions, v_positions) — sorted list of y/x coordinates.
    """
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary mask: dark + colored pixels
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    all_pixels = cv2.bitwise_or(binary, cv2.bitwise_or(mask_r1, mask_r2))
    all_pixels = cv2.bitwise_or(all_pixels, mask_blue)

    # Morphological opening => chỉ giữ đường dài
    h_kl = max(w_img // 15, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kl, 1))
    h_morph = cv2.morphologyEx(all_pixels, cv2.MORPH_OPEN, h_kernel)

    v_kl = max(h_img // 15, 100)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kl))
    v_morph = cv2.morphologyEx(all_pixels, cv2.MORPH_OPEN, v_kernel)

    # Row/col density
    row_density = np.sum(h_morph > 0, axis=1).astype(float)
    col_density = np.sum(v_morph > 0, axis=0).astype(float)

    min_h_span = int(w_img * 0.30)
    min_v_span = int(h_img * 0.20)

    h_cands = np.where(row_density >= min_h_span)[0]
    v_cands = np.where(col_density >= min_v_span)[0]

    def cluster(vals, gap=25):
        if len(vals) == 0:
            return []
        groups = []
        curr = [int(vals[0])]
        for v in vals[1:]:
            v = int(v)
            if v - curr[-1] <= gap:
                curr.append(v)
            else:
                groups.append(curr)
                curr = [v]
        groups.append(curr)
        return groups

    h_groups = cluster(h_cands)
    v_groups = cluster(v_cands)

    h_lines = [int(np.mean(g)) for g in h_groups]
    v_lines = [int(np.mean(g)) for g in v_groups]

    # Loại bỏ đường quá sát mép (< 3% từ mép)
    edge_h = int(h_img * 0.03)
    edge_w = int(w_img * 0.03)
    h_lines = [y for y in h_lines if edge_h < y < h_img - edge_h]
    v_lines = [x for x in v_lines if edge_w < x < w_img - edge_w]

    return sorted(h_lines), sorted(v_lines)


def _merge_close_lines(lines: List[int], min_gap: int) -> List[int]:
    """Merge các đường quá gần nhau (< min_gap pixels)."""
    if not lines:
        return []
    merged = [lines[0]]
    for l in lines[1:]:
        if l - merged[-1] < min_gap:
            # Giữ trung bình
            merged[-1] = (merged[-1] + l) // 2
        else:
            merged.append(l)
    return merged


def split_into_zones(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Chia ảnh thành grid zones dựa trên đường gấp phát hiện được.

    Returns list of:
      {
        "zone_id": int,         # 0-based
        "label": str,           # "R1C1", "R1C2", ...
        "bbox": (x1, y1, x2, y2),
        "image": np.ndarray,    # ảnh zone đã crop
      }
    """
    h_img, w_img = image.shape[:2]
    h_lines, v_lines = _detect_lines(image)

    # Merge đường quá gần nhau
    min_h_gap = max(int(h_img * 0.05), 30)
    min_v_gap = max(int(w_img * 0.05), 30)
    h_lines = _merge_close_lines(h_lines, min_h_gap)
    v_lines = _merge_close_lines(v_lines, min_v_gap)

    logger.info(f"zone_splitter: H-lines={h_lines}, V-lines={v_lines}")

    # Tạo grid boundaries (bao gồm mép ảnh)
    y_bounds = [0] + h_lines + [h_img]
    x_bounds = [0] + v_lines + [w_img]

    zones = []
    zone_id = 0
    for r_idx in range(len(y_bounds) - 1):
        for c_idx in range(len(x_bounds) - 1):
            y1, y2 = y_bounds[r_idx], y_bounds[r_idx + 1]
            x1, x2 = x_bounds[c_idx], x_bounds[c_idx + 1]

            # Bỏ qua zone quá nhỏ (< 5% diện tích ảnh)
            zone_area = (x2 - x1) * (y2 - y1)
            img_area = h_img * w_img
            if zone_area < img_area * 0.02:
                continue

            # Pad inward 3px để tránh bắt đường viền
            pad = 3
            crop_y1 = min(y1 + pad, y2)
            crop_y2 = max(y2 - pad, y1)
            crop_x1 = min(x1 + pad, x2)
            crop_x2 = max(x2 - pad, x1)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            zone_img = image[crop_y1:crop_y2, crop_x1:crop_x2]

            label = f"R{r_idx + 1}C{c_idx + 1}"
            zones.append({
                "zone_id": zone_id,
                "label": label,
                "row": r_idx,
                "col": c_idx,
                "bbox": (x1, y1, x2, y2),
                "image": zone_img,
            })
            zone_id += 1

    logger.info(f"zone_splitter: split into {len(zones)} zones "
                f"(rows={len(y_bounds)-1}, cols={len(x_bounds)-1})")
    return zones


def draw_zones_debug(image: np.ndarray, zones: List[Dict]) -> np.ndarray:
    """Vẽ debug overlay: viền zone + label."""
    vis = image.copy()
    for z in zones:
        x1, y1, x2, y2 = z["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Label
        cv2.putText(vis, z["label"], (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis
