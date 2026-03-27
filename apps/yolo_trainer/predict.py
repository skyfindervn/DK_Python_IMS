"""
predict.py — Dùng mô hình đã train để crop vùng in từ ảnh chụp
================================================================
Input : ảnh chụp thực tế + ảnh maket (để tính Homography sau crop)
Output: ảnh đã crop phẳng, đúng chiều, khớp với maket

Chạy thử:
    python predict.py --image test.jpg --template maket.jpg
    python predict.py --image test.jpg --template maket.jpg --weights weights/carton_seg_best.pt
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "weights/carton_seg_best.pt"
DEFAULT_CONF    = 0.4
MIN_MATCHES     = 15
# ──────────────────────────────────────────────────────────────────────────────


def load_model(weights: str) -> YOLO:
    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy weights: {path}")
    return YOLO(path)


def detect_and_crop(
    model: YOLO,
    image: np.ndarray,
    conf: float = DEFAULT_CONF,
) -> dict:
    """
    Dùng YOLOv8-seg để phát hiện và crop vùng in (polygon 4 góc).

    Returns:
        {
            "status": "success" | "fail",
            "cropped": np.ndarray,       # ảnh đã crop + perspective correct
            "polygon": list[list[int]], # 4 điểm góc pixel (TL,TR,BR,BL)
            "conf": float,
        }
    """
    results = model(image, conf=conf, verbose=False)
    result  = results[0]

    if result.masks is None or len(result.masks) == 0:
        return {"status": "fail", "cropped": image, "polygon": [], "conf": 0.0}

    # Chọn detection có confidence cao nhất
    confs   = result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    # Lấy polygon từ mask
    poly_xy = result.masks.xy[best_idx]  # (N, 2) float32 — tất cả điểm outline

    # Xấp xỉ polygon 4 góc từ contour
    poly_xy = poly_xy.astype(np.float32)
    epsilon  = 0.02 * cv2.arcLength(poly_xy, closed=True)
    approx   = cv2.approxPolyDP(poly_xy, epsilon, closed=True)

    # Nếu không ra đúng 4 điểm → thử convex hull rồi lấy 4 điểm cực đoan
    if len(approx) != 4:
        hull = cv2.convexHull(poly_xy.astype(np.int32))
        approx = _extract_4_corners(hull.reshape(-1, 2))
    else:
        approx = approx.reshape(-1, 2)

    # Sắp xếp TL, TR, BR, BL
    corners = _order_corners(approx.astype(np.float32))

    # Perspective transform thành hình chữ nhật
    w = int(max(
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[3]),
    ))
    h = int(max(
        np.linalg.norm(corners[3] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
    ))

    if w < 50 or h < 50:
        return {"status": "fail", "cropped": image, "polygon": [], "conf": best_conf}

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)

    M        = cv2.getPerspectiveTransform(corners, dst)
    cropped  = cv2.warpPerspective(image, M, (w, h))

    return {
        "status":  "success",
        "cropped": cropped,
        "polygon": corners.tolist(),
        "conf":    best_conf,
    }


def refine_with_template(
    cropped: np.ndarray,
    template: np.ndarray,
) -> dict:
    """
    Dùng SIFT Homography để tinh chỉnh ảnh crop cho khớp pixel-to-pixel với maket.
    """
    max_dim = 1200
    
    def _rs(im):
        h, w = im.shape[:2]
        s = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
        return (cv2.resize(im, (int(w*s), int(h*s))) if s < 1 else im), s

    cropped_small, sc  = _rs(cropped)
    template_small, st = _rs(template)

    g_c = cv2.cvtColor(cropped_small, cv2.COLOR_BGR2GRAY)
    g_t = cv2.cvtColor(template_small, cv2.COLOR_BGR2GRAY)

    sift     = cv2.SIFT_create(nfeatures=3000)
    kp_c, des_c = sift.detectAndCompute(g_c, None)
    kp_t, des_t = sift.detectAndCompute(g_t, None)

    if des_c is None or des_t is None:
        return {"status": "fail", "aligned": cropped, "matches": 0}

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(des_c, des_t, k=2)

    good = [m for m, n in (r for r in raw if len(r) == 2) if m.distance < 0.75 * n.distance]

    if len(good) < MIN_MATCHES:
        return {"status": "fail", "aligned": cropped, "matches": len(good)}

    src = np.float32([kp_c[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if M is None:
        return {"status": "fail", "aligned": cropped, "matches": len(good)}

    inliers = int(np.sum(mask))
    # Scale M về full-resolution
    M_full = np.array([[1/st, 0, 0],[0, 1/st, 0],[0, 0, 1]]) @ M @ np.array([[sc, 0, 0],[0, sc, 0],[0, 0, 1]])
    aligned = cv2.warpPerspective(cropped, M_full, (template.shape[1], template.shape[0]))

    return {"status": "success", "aligned": aligned, "matches": inliers}


def _order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]    # TL
    rect[2] = pts[np.argmax(s)]    # BR
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect


def _extract_4_corners(hull_pts: np.ndarray) -> np.ndarray:
    """Lấy 4 điểm TL, TR, BR, BL cực đoan từ convex hull."""
    cx, cy = hull_pts.mean(axis=0)
    def _quadrant(p):
        return (0 if p[1] < cy else 1, 0 if p[0] < cx else 1)
    corners = {}
    for p in hull_pts:
        q = _quadrant(p)
        key = np.linalg.norm(p - [cx, cy])
        if q not in corners or key > corners[q][0]:
            corners[q] = (key, p)
    pts = np.array([c[1] for c in corners.values()], dtype=np.float32)
    # Nếu thiếu → dùng bounding box
    if len(pts) < 4:
        x, y, w, h = cv2.boundingRect(hull_pts.astype(np.int32))
        pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    image    = cv2.imread(args.image)
    template = cv2.imread(args.template) if args.template else None

    if image is None:
        print(f"❌ Không đọc được ảnh: {args.image}")
        sys.exit(1)

    model  = load_model(args.weights)
    result = detect_and_crop(model, image, conf=args.conf)

    print(f"YOLO detect: status={result['status']}  conf={result['conf']:.2f}")

    if result["status"] != "success":
        print("❌ Không phát hiện được vùng in.")
        sys.exit(2)

    cropped = result["cropped"]
    out_path = args.output or f"output_crop_{Path(args.image).stem}.jpg"
    cv2.imwrite(out_path, cropped)
    print(f"✅ Đã lưu crop: {out_path}")

    if template is not None:
        ref = refine_with_template(cropped, template)
        print(f"SIFT refine : status={ref['status']}  matches={ref['matches']}")
        if ref["status"] == "success":
            out_aligned = f"output_aligned_{Path(args.image).stem}.jpg"
            cv2.imwrite(out_aligned, ref["aligned"])
            print(f"✅ Đã lưu aligned: {out_aligned}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",    required=True)
    ap.add_argument("--template", default=None)
    ap.add_argument("--weights",  default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf",     default=DEFAULT_CONF, type=float)
    ap.add_argument("--output",   default=None)
    main(ap.parse_args())
