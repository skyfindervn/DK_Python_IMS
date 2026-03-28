import cv2
import numpy as np
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ─── YOLO Model singleton (load 1 lần khi service khởi động) ──────────────────
_yolo_model = None
YOLO_WEIGHTS = Path(__file__).parent / "models" / "carton_obb_best.pt"

def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_WEIGHTS.exists():
            logger.warning(f"YOLO weights not found: {YOLO_WEIGHTS}. Skipping YOLO crop.")
            return None
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(str(YOLO_WEIGHTS))
            logger.info(f"YOLO model loaded: {YOLO_WEIGHTS.name}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return None
    return _yolo_model

# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        logger.error(f"Cannot read image: {path}")
    return img


def image_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm góc: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]     # TL
    rect[2] = pts[np.argmax(s)]     # BR
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


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


def yolo_crop(image: np.ndarray, conf: float = 0.35) -> Dict[str, Any]:
    """
    Dùng YOLO OBB để detect và crop vùng in carton.
    Trả về ảnh đã perspective-correct thành hình chữ nhật.
    """
    model = _get_yolo_model()
    if model is None:
        return {"status": "skip", "cropped": image, "conf": 0.0}

    results = model(image, conf=conf, verbose=False)
    result  = results[0]

    if result.obb is None or len(result.obb) == 0:
        logger.warning("YOLO: Không detect được vùng in.")
        return {"status": "fail", "cropped": image, "conf": 0.0}

    # Chọn detection confidence cao nhất
    confs     = result.obb.conf.cpu().numpy()
    best_idx  = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    # Lấy 4 góc OBB (xyxyxyxy format) → shape (4,2)
    corners_xy = result.obb.xyxyxyxy[best_idx].cpu().numpy().reshape(4, 2)
    corners    = _order_corners(corners_xy.astype(np.float32))

    # ── Validate trước khi transform ──────────────────────────────────────────
    h_img, w_img = image.shape[:2]
    if not _validate_quad(corners, h_img, w_img):
        logger.warning(f"YOLO: Góc detect không hợp lệ (degenerate quad). Bỏ qua.")
        return {"status": "fail", "cropped": image, "conf": best_conf}
    # ──────────────────────────────────────────────────────────────────────────

    # Tính kích thước output
    w = int(max(
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[3]),
    ))
    h = int(max(
        np.linalg.norm(corners[3] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
    ))

    if w < 50 or h < 50:
        logger.warning(f"YOLO: Crop quá nhỏ ({w}x{h}). Bỏ qua.")
        return {"status": "fail", "cropped": image, "conf": best_conf}

    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    M       = cv2.getPerspectiveTransform(corners, dst)
    cropped = cv2.warpPerspective(image, M, (w, h))

    logger.info(f"YOLO crop OK: conf={best_conf:.2f}, size={w}x{h}px")
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

    if crop_result["status"] == "success":
        img_to_align = crop_result["cropped"]
        logger.info(f"YOLO crop thành công, tiến hành SIFT align.")
    else:
        img_to_align = image
        logger.info("YOLO không crop được, dùng ảnh gốc cho SIFT align.")

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
