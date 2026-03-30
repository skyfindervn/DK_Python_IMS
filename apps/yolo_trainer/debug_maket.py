"""
V9: Intersection-based scoring

Key insight: Carton border lines have MANY INTERSECTIONS with perpendicular lines
(internal dividers/fold lines), while dimension annotation lines have
FEW OR NO intersections.

Strategy:
  1. Detect H and V morphological lines
  2. For each H-line, count how many V-lines intersect it
  3. For each V-line, count how many H-lines intersect it
  4. Lines with more intersections are more likely to be real borders
  5. Find outermost rectangle using only high-intersection lines
"""
import cv2
import numpy as np
from pathlib import Path


def crop_maket_by_border(image: np.ndarray, debug_name: str = "") -> dict:
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # === Step 1: Create masks ===
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv, np.array([0, 50, 50]),   np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    all_pixels = cv2.bitwise_or(binary, cv2.bitwise_or(mask_r1, mask_r2))
    all_pixels = cv2.bitwise_or(all_pixels, mask_blue)
    
    # Extract H lines
    h_kl = max(w_img // 15, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kl, 1))
    h_morph = cv2.morphologyEx(all_pixels, cv2.MORPH_OPEN, h_kernel)
    
    # Extract V lines
    v_kl = max(h_img // 15, 100)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kl))
    v_morph = cv2.morphologyEx(all_pixels, cv2.MORPH_OPEN, v_kernel)
    
    # === Step 2: Find line positions ===
    row_density = np.sum(h_morph > 0, axis=1).astype(float)
    col_density = np.sum(v_morph > 0, axis=0).astype(float)
    
    min_h_span = int(w_img * 0.35)
    min_v_span = int(h_img * 0.25)
    
    h_cands = np.where(row_density >= min_h_span)[0]
    v_cands = np.where(col_density >= min_v_span)[0]
    
    def cluster(vals, gap=25):
        if len(vals) == 0: return []
        groups = []
        curr = [int(vals[0])]
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
    
    # Filter edge lines
    edge_h = int(h_img * 0.04)
    edge_w = int(w_img * 0.04)
    h_lines = [l for l in h_lines if edge_h < l[0] < h_img - edge_h]
    v_lines = [l for l in v_lines if edge_w < l[0] < w_img - edge_w]
    
    print(f"  H-lines: {[(l[0], f'{l[1]:.0f}') for l in h_lines]}")
    print(f"  V-lines: {[(l[0], f'{l[1]:.0f}') for l in v_lines]}")
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        return {"status": "fail", "cropped": image}
    
    # === Step 3: Count intersections for each line ===
    # An intersection exists where both h_morph and v_morph have pixels
    intersection_map = cv2.bitwise_and(
        cv2.dilate(h_morph, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))),
        cv2.dilate(v_morph, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    )
    
    # For each H-line, count intersections with V-lines
    h_intersections = {}
    for hy, _ in h_lines:
        count = 0
        for vx, _ in v_lines:
            # Check intersection point
            patch = intersection_map[max(0,hy-15):min(h_img,hy+16), max(0,vx-15):min(w_img,vx+16)]
            if np.sum(patch > 0) > 5:
                count += 1
        h_intersections[hy] = count
    
    # For each V-line, count intersections with H-lines
    v_intersections = {}
    for vx, _ in v_lines:
        count = 0
        for hy, _ in h_lines:
            patch = intersection_map[max(0,hy-15):min(h_img,hy+16), max(0,vx-15):min(w_img,vx+16)]
            if np.sum(patch > 0) > 5:
                count += 1
        v_intersections[vx] = count
    
    print(f"  H intersections: {h_intersections}")
    print(f"  V intersections: {v_intersections}")
    
    # === Step 4: Filter lines by intersection count ===
    # Lines with >= 2 intersections are likely border lines
    max_h_int = max(h_intersections.values()) if h_intersections else 0
    max_v_int = max(v_intersections.values()) if v_intersections else 0
    
    # Threshold: at least 50% of max intersections, minimum 2
    h_threshold = max(2, max_h_int * 0.4)
    v_threshold = max(2, max_v_int * 0.4)
    
    h_border = [(y, d) for y, d in h_lines if h_intersections.get(y, 0) >= h_threshold]
    v_border = [(x, d) for x, d in v_lines if v_intersections.get(x, 0) >= v_threshold]
    
    print(f"  Border H (>={h_threshold:.0f} int): {[l[0] for l in h_border]}")
    print(f"  Border V (>={v_threshold:.0f} int): {[l[0] for l in v_border]}")
    
    if len(h_border) < 2 or len(v_border) < 2:
        # Fallback: use top intersection lines
        h_border = sorted(h_lines, key=lambda l: h_intersections.get(l[0], 0), reverse=True)[:max(2, len(h_lines))]
        v_border = sorted(v_lines, key=lambda l: v_intersections.get(l[0], 0), reverse=True)[:max(2, len(v_lines))]
        print(f"  Fallback border H: {[l[0] for l in h_border]}")
        print(f"  Fallback border V: {[l[0] for l in v_border]}")
    
    # === Step 5: Find the outermost rectangle from border lines ===
    h_positions = sorted([l[0] for l in h_border])
    v_positions = sorted([l[0] for l in v_border])
    
    # Outermost = first and last positions
    yt = h_positions[0]
    yb = h_positions[-1]
    xl = v_positions[0]
    xr = v_positions[-1]
    
    print(f"  Rectangle: ({xl},{yt})->({xr},{yb})")
    
    # Validate
    if xr - xl < w_img * 0.3 or yb - yt < h_img * 0.3:
        return {"status": "fail", "cropped": image}
    
    # Pad inward
    pad = 5
    yt2, yb2 = yt + pad, yb - pad
    xl2, xr2 = xl + pad, xr - pad
    
    cropped = image[yt2:yb2, xl2:xr2]
    
    if debug_name:
        vis = image.copy()
        # Draw all lines (thin blue)
        for l in h_lines:
            cv2.line(vis, (0, l[0]), (w_img, l[0]), (255, 100, 0), 1)
        for l in v_lines:
            cv2.line(vis, (l[0], 0), (l[0], h_img), (255, 100, 0), 1)
        # Draw border lines (thick cyan)
        for l in h_border:
            cv2.line(vis, (0, l[0]), (w_img, l[0]), (255, 255, 0), 2)
        for l in v_border:
            cv2.line(vis, (l[0], 0), (l[0], h_img), (255, 255, 0), 2)
        # Draw result rectangle (green)
        cv2.rectangle(vis, (xl, yt), (xr, yb), (0, 255, 0), 4)
        
        # Draw intersection points
        for hy, _ in h_lines:
            for vx, _ in v_lines:
                patch = intersection_map[max(0,hy-15):min(h_img,hy+16), max(0,vx-15):min(w_img,vx+16)]
                if np.sum(patch > 0) > 5:
                    cv2.circle(vis, (vx, hy), 12, (0, 0, 255), 3)
        
        cv2.imwrite(f"debug_{debug_name}_box.jpg", vis)
        cv2.imwrite(f"debug_{debug_name}_crop.jpg", cropped)
    
    return {"status": "success", "cropped": cropped}


markets_dir = Path(r"d:\onedriver\OneDrive\Bao_bi_carton\Github\DK2IMS\public\upload\markets")
for f in sorted(markets_dir.glob("*.jpg")):
    print(f"\n{'='*60}")
    print(f"File: {f.name}")
    img = cv2.imread(str(f))
    print(f"  Original: {img.shape[1]}x{img.shape[0]}")
    result = crop_maket_by_border(img, debug_name=f.stem)
    print(f"  Status: {result['status']}")
    if result["status"] == "success":
        h, w = result["cropped"].shape[:2]
        print(f"  Cropped: {w}x{h}")
