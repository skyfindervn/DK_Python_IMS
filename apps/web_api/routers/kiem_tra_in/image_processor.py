import cv2
import numpy as np
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import log_broker

logger = logging.getLogger(__name__)

# ─── Helper: push ảnh debug lên UI (resize nhỏ để tránh tắc SSE) ─────────────
def _push_debug_image(title: str, img: np.ndarray, step: str) -> None:
    """Encode ảnh → base64 và push qua log_broker với event_type='debug_image'."""
    try:
        h, w = img.shape[:2]
        # Resize nếu ảnh quá lớn (max 800px cạnh dài)
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf).decode()
        log_broker.push(
            title,
            level="INFO",
            event_type="debug_image",
            step=step,
            base64=b64,
        )
    except Exception as e:
        logger.warning(f"_push_debug_image failed: {e}")

# ─── YOLO Model singleton (load lazily theo tên file) ───────────────────────────
_yolo_models = {}

def _get_yolo_model(weight_name="findcarton_seg_best.pt"):
    global _yolo_models
    if weight_name not in _yolo_models:
        weight_path = Path(__file__).parent.parent.parent.parent / "yolo_trainer" / weight_name
        if not weight_path.exists():
            logger.warning(f"YOLO weights not found: {weight_path}. Skipping YOLO crop.")
            _yolo_models[weight_name] = None
        else:
            try:
                from ultralytics import YOLO
                _yolo_models[weight_name] = YOLO(str(weight_path))
                logger.info(f"YOLO model loaded: {weight_path.name}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model {weight_name}: {e}")
                _yolo_models[weight_name] = None
    return _yolo_models[weight_name]

# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        logger.error(f"Cannot read image: {path}")
    return img


def image_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _order_corners(box: np.ndarray) -> np.ndarray:
    """
    Sap xep 4 diem thanh TL, TR, BR, BL (chieu kim dong ho).
    Dung centroid + left/right split - dang tin hon sum/diff khi anh meo.
    """
    box = box.reshape(4, 2).astype(np.float32)
    cx, cy = box.mean(axis=0)

    left_pts = box[box[:, 0] < cx]
    right_pts = box[box[:, 0] >= cx]

    if len(left_pts) == 2 and len(right_pts) == 2:
        if left_pts[0][1] < left_pts[1][1]:
            tl, bl = left_pts[0], left_pts[1]
        else:
            tl, bl = left_pts[1], left_pts[0]
        if right_pts[0][1] < right_pts[1][1]:
            tr, br = right_pts[0], right_pts[1]
        else:
            tr, br = right_pts[1], right_pts[0]
    else:
        s = box.sum(axis=1)
        d = np.diff(box, axis=1).ravel()
        tl = box[np.argmin(s)]
        tr = box[np.argmax(d)]
        br = box[np.argmax(s)]
        bl = box[np.argmin(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _validate_quad(corners: np.ndarray, img_h: int, img_w: int) -> bool:
    """
    Kiểm tra 4 góc có tạo thành tứ giác hợp lệ không.
    Từ chối nếu:
    - Diện tích quá nhỏ (< 5% ảnh gốc)
    - Các điểm gần thẳng hàng → homography suy biến (degenerate)
    """
    # Diện tích tứ giác bằng công thức Shoelace
    pts = corners.reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(
        x[0]*y[1] - x[1]*y[0] +
        x[1]*y[2] - x[2]*y[1] +
        x[2]*y[3] - x[3]*y[2] +
        x[3]*y[0] - x[0]*y[3]
    )
    img_area = img_h * img_w
    if area < 0.05 * img_area:
        logger.warning(f"YOLO quad validation fail: area={area:.0f} < 5% of image ({0.05*img_area:.0f})")
        return False

    # Kiểm tra không suy biến: các cạnh phải có độ dài tối thiểu
    min_side = min(
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
        np.linalg.norm(corners[3] - corners[2]),
        np.linalg.norm(corners[0] - corners[3]),
    )
    if min_side < 20:
        logger.warning(f"YOLO quad validation fail: min_side={min_side:.1f}px too small")
        return False

    return True


def _extract_homography_rotation(M: Optional[np.ndarray]) -> Optional[float]:
    """
    Trích xuất góc xoay (degrees) từ ma trận homography 3×3.
    Ảnh đúng chiều → ~0°, ảnh ngược 180° → ~±180°.
    Dùng xấp xỉ affine từ 2×2 submatrix.
    """
    if M is None:
        return None
    angle1 = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    angle2 = np.degrees(np.arctan2(-M[0, 1], M[1, 1]))
    return (angle1 + angle2) / 2.0


def crop_maket_by_border(image: np.ndarray) -> Dict[str, Any]:
    """
    Crop vùng thiết kế chính trên maket carton bằng cách phát hiện đường viền
    dùng morphological line detection + intersection scoring.
    
    Thuật toán:
      1. Trích xuất đường ngang/dọc bằng morphological opening
      2. Đếm số giao điểm (intersection) của mỗi đường với đường vuông góc
      3. Đường viền carton thực có NHIỀU giao điểm (giao với fold lines bên trong)
         trong khi đường dimension annotation có ÍT hoặc KHÔNG giao điểm
      4. Chọn hình chữ nhật ngoài cùng từ các đường có intersection cao
    
    Returns dict: {"status": "success"/"fail", "cropped": np.ndarray}
    """
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # === Step 1: Extract all dark + colored pixels ===
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv, np.array([0, 50, 50]),   np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    all_pixels = cv2.bitwise_or(binary, cv2.bitwise_or(mask_r1, mask_r2))
    all_pixels = cv2.bitwise_or(all_pixels, mask_blue)
    
    # === Step 2: Extract horizontal and vertical lines via morphology ===
    h_kl = max(w_img // 15, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kl, 1))
    h_morph = cv2.morphologyEx(all_pixels, cv2.MORPH_OPEN, h_kernel)
    
    v_kl = max(h_img // 15, 100)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kl))
    v_morph = cv2.morphologyEx(all_pixels, cv2.MORPH_OPEN, v_kernel)
    
    # === Step 3: Find line positions by row/col density ===
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
    
    # Filter edge lines (4% margin)
    edge_h = int(h_img * 0.04)
    edge_w = int(w_img * 0.04)
    h_lines = [l for l in h_lines if edge_h < l[0] < h_img - edge_h]
    v_lines = [l for l in v_lines if edge_w < l[0] < w_img - edge_w]
    
    logger.debug(f"crop_maket_by_border: H-lines={[l[0] for l in h_lines]}, V-lines={[l[0] for l in v_lines]}")
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        logger.warning(f"crop_maket_by_border: not enough lines H={len(h_lines)} V={len(v_lines)}")
        return {"status": "fail", "cropped": image}
    
    # === Step 4: Count intersections for each line ===
    intersection_map = cv2.bitwise_and(
        cv2.dilate(h_morph, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))),
        cv2.dilate(v_morph, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    )
    
    h_intersections = {}
    for hy, _ in h_lines:
        count = 0
        for vx, _ in v_lines:
            patch = intersection_map[max(0,hy-15):min(h_img,hy+16), max(0,vx-15):min(w_img,vx+16)]
            if np.sum(patch > 0) > 5:
                count += 1
        h_intersections[hy] = count
    
    v_intersections = {}
    for vx, _ in v_lines:
        count = 0
        for hy, _ in h_lines:
            patch = intersection_map[max(0,hy-15):min(h_img,hy+16), max(0,vx-15):min(w_img,vx+16)]
            if np.sum(patch > 0) > 5:
                count += 1
        v_intersections[vx] = count
    
    logger.debug(f"crop_maket_by_border: H_int={h_intersections}, V_int={v_intersections}")
    
    # === Step 5: Filter lines by intersection count ===
    max_h_int = max(h_intersections.values()) if h_intersections else 0
    max_v_int = max(v_intersections.values()) if v_intersections else 0
    
    h_threshold = max(2, max_h_int * 0.4)
    v_threshold = max(2, max_v_int * 0.4)
    
    h_border = [(y, d) for y, d in h_lines if h_intersections.get(y, 0) >= h_threshold]
    v_border = [(x, d) for x, d in v_lines if v_intersections.get(x, 0) >= v_threshold]
    
    if len(h_border) < 2 or len(v_border) < 2:
        # Fallback: use lines with most intersections
        h_border = sorted(h_lines, key=lambda l: h_intersections.get(l[0], 0), reverse=True)
        v_border = sorted(v_lines, key=lambda l: v_intersections.get(l[0], 0), reverse=True)
    
    # === Step 6: Outermost rectangle from border lines ===
    h_positions = sorted([l[0] for l in h_border])
    v_positions = sorted([l[0] for l in v_border])
    
    yt, yb = h_positions[0], h_positions[-1]
    xl, xr = v_positions[0], v_positions[-1]
    
    if xr - xl < w_img * 0.3 or yb - yt < h_img * 0.3:
        logger.warning(f"crop_maket_by_border: rect too small ({xr-xl}x{yb-yt})")
        return {"status": "fail", "cropped": image}
    
    # Pad inward to exclude the border line itself
    pad = 5
    yt2, yb2 = yt + pad, yb - pad
    xl2, xr2 = xl + pad, xr - pad
    
    cropped = image[yt2:yb2, xl2:xr2]
    logger.info(f"Maket border crop: ({xl},{yt})->({xr},{yb}), {xr-xl}x{yb-yt}")
    return {"status": "success", "cropped": cropped}

def yolo_crop(image: np.ndarray, conf: float = 0.35, model_name: str = "findcarton_seg_best.pt") -> Dict[str, Any]:
    """
    Dùng YOLO Segmentation để detect và crop vùng in carton.
    Pipeline:
      1. Chạy model seg → lấy mask của detection cao nhất
      2. Resize mask về kích thước ảnh gốc → tìm contour lớn nhất
      3. approxPolyDP → tứ giác 4 điểm
      4. Perspective transform → ảnh crop thẳng đứng
    """
    model = _get_yolo_model(model_name)
    if model is None:
        return {"status": "skip", "cropped": image, "conf": 0.0}

    results = model(image, conf=conf, verbose=False)
    result  = results[0]

    # ── Debug Step 1: ảnh annotated với boxes ─────────────────────────────────
    vis_detect = result.plot()  # YOLO tự vẽ boxes + masks
    _push_debug_image("🔍 [YOLO-1] Kết quả detect", vis_detect, "yolo_detect")

    # Kiểm tra có detection không
    if len(result.boxes) == 0:
        logger.warning("YOLO: Không detect được object nào.")
        return {"status": "fail", "cropped": image, "conf": 0.0}

    confs     = result.boxes.conf.cpu().numpy()
    best_conf = float(np.max(confs))
    n_det     = len(confs)
    h_img, w_img = image.shape[:2]
    mask_bin = np.zeros((h_img, w_img), dtype=np.uint8)

    # Nếu là mô hình Segmentation (có masks)
    if result.masks is not None and len(result.masks) > 0:
        for i in range(n_det):
            m = result.masks.data[i].cpu().numpy()
            m_uint8 = (m * 255).astype(np.uint8)
            m_full  = cv2.resize(m_uint8, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            mask_bin = cv2.bitwise_or(mask_bin, (m_full > 127).astype(np.uint8) * 255)
        logger.info(f"YOLO Seg: merged {n_det} masks")
    # Nếu là mô hình Detection/OBB thuần (chỉ có box, không có masks)
    else:
        for i in range(n_det):
            box = result.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            cv2.rectangle(mask_bin, (x1, y1), (x2, y2), 255, -1)
        logger.info(f"YOLO Detect: merged {n_det} bounding boxes as mask")

    # ── Debug Step 2: mask overlay ────────────────────────────────────────────
    overlay = image.copy()
    overlay[mask_bin > 0] = (
        overlay[mask_bin > 0] * 0.45 + np.array([0, 180, 0]) * 0.55
    ).astype(np.uint8)
    mask_cov = round(np.sum(mask_bin > 0) / (w_img * h_img) * 100, 1)
    _push_debug_image(
        f"🎭 [YOLO-2] Mask ({n_det} det merged) | coverage={mask_cov}% | conf={best_conf:.2f}",
        overlay, "yolo_mask"
    )

    # Tìm contour lớn nhất trong mask
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("YOLO Seg: Không tìm được contour trong mask.")
        return {"status": "fail", "cropped": image, "conf": best_conf}

    largest_cnt = max(contours, key=cv2.contourArea)

    # LUÔN dùng minAreaRect trực tiếp trên contour (bỏ approxPolyDP — không ổn định)
    rect = cv2.minAreaRect(largest_cnt)
    box = cv2.boxPoints(rect)
    corners_xy = box.astype(np.float32)

    corners = _order_corners(corners_xy)

    # Đảm bảo output là landscape (rộng > cao) — carton luôn nằm ngang
    w_out = int(max(
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[3]),
    ))
    h_out = int(max(
        np.linalg.norm(corners[3] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
    ))
    if h_out > w_out:
        # Xoay thứ tự góc 90° để thành landscape: TL←BL, TR←TL, BR←TR, BL←BR
        corners = np.array([corners[3], corners[0], corners[1], corners[2]], dtype=np.float32)
        w_out, h_out = h_out, w_out

    # ── Debug Step 3: corners overlay ─────────────────────────────────────────
    vis_corners = image.copy()
    cv2.drawContours(vis_corners, [largest_cnt], -1, (255, 140, 0), 2)
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0,255,0), (0,0,255), (255,0,255), (255,165,0)]
    for i, (pt, lbl, col) in enumerate(zip(corners, labels, colors)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis_corners, (x, y), 10, col, -1)
        cv2.putText(vis_corners, lbl, (x+5, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.polylines(vis_corners,
                  [corners.astype(np.int32).reshape(-1, 1, 2)],
                  True, (0, 255, 255), 3)
    _push_debug_image(
        f"📐 [YOLO-3] Góc (minAreaRect) | {w_out}×{h_out}px",
        vis_corners, "yolo_corners"
    )

    # Validate tứ giác
    if not _validate_quad(corners, h_img, w_img):
        logger.warning("YOLO Seg: Tứ giác detect không hợp lệ. Bỏ qua.")
        return {"status": "fail", "cropped": image, "conf": best_conf}

    # Dùng w_out, h_out đã tính từ trên
    w, h = w_out, h_out

    if w < 50 or h < 50:
        logger.warning(f"YOLO Seg: Crop quá nhỏ ({w}x{h}). Bỏ qua.")
        return {"status": "fail", "cropped": image, "conf": best_conf}

    dst     = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    M       = cv2.getPerspectiveTransform(corners, dst)
    
    # ── Warp cả ảnh sắc nét và mask ───────────────────────────────────────────
    cropped_img  = cv2.warpPerspective(image, M, (w, h))
    cropped_mask = cv2.warpPerspective(mask_bin, M, (w, h))
    
    # Làm mờ mạnh (blur = 101x101) để triệt tiêu hoàn toàn chữ ở background
    blurred_img = cv2.GaussianBlur(cropped_img, (101, 101), 0)
    # Thu hẹp mask một chút để cắt viền êm hơn, rồi làm nhòe viền mask (feather)
    mask_soft = cv2.GaussianBlur(cropped_mask, (21, 21), 0)
    alpha = mask_soft.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    # Trộn ảnh khu vực mask (rõ nét) với mảng ngoài mask (nhòe)
    cropped = (cropped_img * alpha + blurred_img * (1.0 - alpha)).astype(np.uint8)

    # Sanity check: kết quả không được đen quá nhiều (>60% pixel là màu đen = warp sai)
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    black_ratio = np.sum(gray_crop < 10) / gray_crop.size
    if black_ratio > 0.6:
        logger.warning(f"YOLO Seg: Warp kết quả đen {black_ratio:.0%} → góc sai, bỏ qua.")
        _push_debug_image(
            f"❌ [YOLO-4] Warp FAIL — đen {black_ratio:.0%} → góc sai thứ tự",
            cropped, "yolo_crop_fail"
        )
        return {"status": "fail", "cropped": image, "conf": best_conf}

    # ── Debug Step 4: crop cuối ────────────────────────────────────────────────
    _push_debug_image(
        f"✅ [YOLO-4] Crop OK | {w}×{h}px | đen={black_ratio:.0%}",
        cropped, "yolo_crop_ok"
    )
    logger.info(f"YOLO Seg crop OK: conf={best_conf:.2f}, size={w}x{h}px, black={black_ratio:.0%}")
    return {"status": "success", "cropped": cropped, "conf": best_conf}


def universal_align(image: np.ndarray, template: np.ndarray) -> Dict[str, Any]:
    """
    Pipeline 2 bước:
      1. YOLO OBB → crop vùng in (nếu có weights)
      2. SIFT + RANSAC Homography → căn chỉnh pixel với maket

    Nếu YOLO không có weights hoặc fail → dùng SIFT trực tiếp trên ảnh gốc.
    """
    if image is None or template is None:
        return {"status": "fail", "aligned_image": None, "match_count": 0, "yolo_conf": 0.0}

    # ── Bước 1: YOLO Crop ──────────────────────────────────────────────────────
    crop_result = yolo_crop(image)
    yolo_conf   = crop_result["conf"]

    # Bước 2: Auto-rotate 180 độ nếu chữ bị ngược (SIFT Inliers Heuristic)
    # Lấy SIFT features của maket (template)
    max_dim = 1500
    
    def get_resized_and_scales(im):
        h, w = im.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            return cv2.resize(im, (int(w * scale), int(h * scale))), scale, scale
        return im, 1.0, 1.0

    tpl_resized, scale_x_t, scale_y_t = get_resized_and_scales(template)
    gray_tpl = cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2GRAY) if len(tpl_resized.shape) == 3 else tpl_resized
    sift = cv2.SIFT_create(nfeatures=2000)
    kp_t, des_t = sift.detectAndCompute(gray_tpl, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def count_sift_inliers(img_cand: np.ndarray) -> Tuple[int, Optional[float]]:
        """Trả về (inliers, rotation_deg). rotation_deg = góc xoay homography."""
        if des_t is None: return (0, None)
        cand_res, _, _ = get_resized_and_scales(img_cand)
        gray_cand = cv2.cvtColor(cand_res, cv2.COLOR_BGR2GRAY) if len(cand_res.shape) == 3 else cand_res
        kp_i, des_i = sift.detectAndCompute(gray_cand, None)
        if des_i is None or len(des_i) < 2: return (0, None)
        try:
            raw_matches = bf.knnMatch(des_i, des_t, k=2)
            good_matches = [m for m, n in (r for r in raw_matches if len(r) == 2) if m.distance < 0.75 * n.distance]
            if len(good_matches) < 10:
                return (len(good_matches), None)
            src_pts = np.float32([kp_i[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(np.sum(mask)) if mask is not None else 0
            rot_deg = _extract_homography_rotation(M)
            return (inliers, rot_deg)
        except Exception:
            return (0, None)


    if crop_result["status"] == "success":
        # YOLO crop đã perspective-transform thành công
        cropped = crop_result["cropped"]

        # ── Step 5: Xác định hướng ảnh bằng orientation_detector ────────
        # YOLO crop perspective transform có thể tạo ảnh bị:
        #   - xoay 180° (rot180)
        #   - lật ngang/mirror (flip_h)
        #   - lật dọc (flip_v)
        # Dùng ORB Feature Matching + RANSAC inliers so sánh 4 variants với maket
        from .orientation_detector import detect_orientation

        orient_result = detect_orientation(cropped, template)
        best_name  = orient_result.get("orientation", "original")
        best_score = orient_result.get("score", 0)
        best_img   = orient_result.get("image", cropped)

        logger.info(f"Orientation: best='{best_name}' score={best_score}")

        if best_name != "original" and best_score >= 4:
            cropped = best_img
            _push_debug_image(
                f"🔄 [STEP 5] {best_name} | score={best_score}",
                cropped, "yolo_rotate"
            )
        else:
            _push_debug_image(
                f"✅ [STEP 5] original | score={best_score}",
                cropped, "yolo_rotate"
            )

        h_tpl, w_tpl = template.shape[:2]
        aligned = cv2.resize(cropped, (w_tpl, h_tpl))
        logger.info(f"YOLO crop OK → {best_name} → resize {cropped.shape[1]}×{cropped.shape[0]} → {w_tpl}×{h_tpl}")
        return {
            "status": "success",
            "aligned_image": aligned,
            "match_count": best_score,
            "yolo_conf": yolo_conf,
        }

    # YOLO crop fail → dùng SIFT + RANSAC trên ảnh gốc
    img_to_align = image
    logger.info("YOLO không crop được, dùng ảnh gốc cho SIFT align toàn diện.")

    # ── Bước 2: SIFT + RANSAC Homography ───────────────────────────────────────
    max_dim = 1500

    def get_resized_and_scales(im):
        h, w = im.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            return cv2.resize(im, (int(w * scale), int(h * scale))), scale, scale
        return im, 1.0, 1.0

    tpl_resized, scale_x_t, scale_y_t = get_resized_and_scales(template)
    gray_tpl = cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2GRAY) if len(tpl_resized.shape) == 3 else tpl_resized

    sift = cv2.SIFT_create(nfeatures=5000)
    kp_t, des_t = sift.detectAndCompute(gray_tpl, None)

    if des_t is None:
        return {"status": "fail", "aligned_image": img_to_align, "match_count": 0, "yolo_conf": yolo_conf}

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Test cả ảnh gốc + ảnh lật ngang
    variants = {"original": img_to_align, "flipped": cv2.flip(img_to_align, 1)}

    best_status   = "fail"
    best_img      = cv2.resize(img_to_align, (template.shape[1], template.shape[0]))
    best_matches  = 0

    for name, img_cand in variants.items():
        img_resized, scale_x_i, scale_y_i = get_resized_and_scales(img_cand)
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if len(img_resized.shape) == 3 else img_resized

        kp_i, des_i = sift.detectAndCompute(gray_img, None)
        if des_i is None:
            continue

        try:
            raw_matches = bf.knnMatch(des_i, des_t, k=2)
        except Exception:
            continue

        good = [m for m, n in (r for r in raw_matches if len(r) == 2) if m.distance < 0.75 * n.distance]

        if len(good) < 10:
            continue

        src_pts = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M_resized, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M_resized is None:
            continue

        inliers = int(np.sum(mask)) if mask is not None else 0
        if inliers > best_matches:
            # Scale homography về full-resolution
            S_img     = np.diag([scale_x_i, scale_y_i, 1.0])
            S_tpl_inv = np.diag([1.0/scale_x_t, 1.0/scale_y_t, 1.0])
            M_full    = S_tpl_inv @ M_resized @ S_img

            # ── Validate homography không suy biến ────────────────────────────
            # 1. Kiểm tra det(M) ≠ 0 và không quá lớn/nhỏ
            det = np.linalg.det(M_full[:2, :2])
            if abs(det) < 1e-4 or abs(det) > 1e4:
                logger.warning(f"SIFT [{name}]: Homography suy biến (det={det:.4f}). Bỏ qua.")
                continue

            # 2. Kiểm tra 4 góc ảnh input project vào vùng hợp lý của template
            h_tpl, w_tpl = template.shape[:2]
            h_cand, w_cand = img_cand.shape[:2]
            src_corners = np.float32([
                [0, 0], [w_cand, 0], [w_cand, h_cand], [0, h_cand]
            ]).reshape(-1, 1, 2)
            dst_corners = cv2.perspectiveTransform(src_corners, M_full).reshape(-1, 2)
            
            # Các góc warp phải nằm trong khoảng [-50%, 150%] kích thước template
            margin = 0.5
            if (np.any(dst_corners[:, 0] < -w_tpl * margin) or
                np.any(dst_corners[:, 0] > w_tpl * (1 + margin)) or
                np.any(dst_corners[:, 1] < -h_tpl * margin) or
                np.any(dst_corners[:, 1] > h_tpl * (1 + margin))):
                logger.warning(f"SIFT [{name}]: Góc warp ra ngoài vùng hợp lệ. Bỏ qua.")
                continue
            # ──────────────────────────────────────────────────────────────────

            best_matches = inliers
            best_status  = "success"
            h, w         = template.shape[:2]
            best_img     = cv2.warpPerspective(img_cand, M_full, (w, h))

    logger.info(f"Universal align: status={best_status}, inliers={best_matches}, yolo_conf={yolo_conf:.2f}")
    return {
        "status":        best_status,
        "aligned_image": best_img,
        "match_count":   best_matches,
        "yolo_conf":     yolo_conf,
    }
