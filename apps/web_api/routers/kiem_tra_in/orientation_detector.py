import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def detect_orientation(image: np.ndarray, template: np.ndarray) -> dict:
    """
    Phát hiện hướng ảnh bằng SIFT Feature Matching + RANSAC.
    So sánh 4 biến thể (original, rot180, flip_h, flip_v) với template (maket).
    Trả về variant có nhiều RANSAC inliers nhất.
    """
    sift = cv2.SIFT_create(nfeatures=3000)

    # Resize template để tăng tốc
    h_t, w_t = template.shape[:2]
    if max(h_t, w_t) > 1200:
        scale_t = 1200 / max(h_t, w_t)
        tmp = cv2.resize(template, (int(w_t * scale_t), int(h_t * scale_t)))
    else:
        tmp = template.copy()

    gray_t = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY) if len(tmp.shape) == 3 else tmp
    kp_t, des_t = sift.detectAndCompute(gray_t, None)

    if des_t is None:
        logger.warning("Orientation: No features found in template.")
        return {"orientation": "original", "score": 0, "image": image}

    candidates = {
        "original": image,
        "rot180": cv2.rotate(image, cv2.ROTATE_180),
        "flip_h": cv2.flip(image, 1),
        "flip_v": cv2.flip(image, 0)
    }

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    best_name = "original"
    best_score = -1
    best_img = image
    all_scores = {}

    for name, cand in candidates.items():
        h_c, w_c = cand.shape[:2]
        if max(h_c, w_c) > 1200:
            scale_c = 1200 / max(h_c, w_c)
            cand_resized = cv2.resize(cand, (int(w_c * scale_c), int(h_c * scale_c)))
        else:
            cand_resized = cand.copy()

        gray_c = cv2.cvtColor(cand_resized, cv2.COLOR_BGR2GRAY) if len(cand_resized.shape) == 3 else cand_resized
        kp_c, des_c = sift.detectAndCompute(gray_c, None)

        if des_c is None or len(des_c) < 10:
            all_scores[name] = 0
            continue

        try:
            matches = bf.knnMatch(des_c, des_t, k=2)
        except Exception:
            all_scores[name] = 0
            continue

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Dùng RANSAC + kiểm tra góc xoay homography
        if len(good_matches) >= 8:
            src_pts = np.float32([kp_c[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(np.sum(mask)) if mask is not None else 0

            # Bonus: nếu homography rotation ≈ 0° → ảnh đúng chiều → điểm cộng
            rot_bonus = 0
            if M is not None:
                angle1 = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
                angle2 = np.degrees(np.arctan2(-M[0, 1], M[1, 1]))
                rot_deg = (angle1 + angle2) / 2.0
                abs_rot = abs(rot_deg)
                if abs_rot > 180:
                    abs_rot = 360 - abs_rot
                # H rotation gần 0° = ảnh match tốt không cần xoay thêm
                if abs_rot < 30:
                    rot_bonus = inliers * 0.5  # Bonus 50%
                # H rotation gần 180° = ảnh đang ngược → trừ điểm
                elif abs_rot > 150:
                    rot_bonus = -inliers * 0.3
                logger.debug(f"  [{name}] H_rot={rot_deg:.1f}° → bonus={rot_bonus:.0f}")

            score = inliers + rot_bonus
        else:
            score = len(good_matches) * 0.5  # Ít matches → điểm thấp

        all_scores[name] = round(score, 1)

        if score > best_score:
            best_score = score
            best_name = name
            best_img = cand

    logger.info(f"Orientation: {all_scores} → best='{best_name}' score={best_score}")
    return {"orientation": best_name, "score": best_score, "image": best_img}
