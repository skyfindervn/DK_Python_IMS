import cv2
import numpy as np
import logging
import base64
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        logger.error(f"Cannot read image: {path}")
    return img

def image_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()

def universal_align(image: np.ndarray, template: np.ndarray) -> Dict[str, Any]:
    """
    Universal image alignment without edge detection.
    Directly aligns input image to reference template using SIFT + RANSAC homography.
    Handles affine transforms, perspective distortion, rotation, and flipping.
    """
    if image is None or template is None:
        return {"status": "fail", "aligned_image": None, "match_count": 0}

    max_dim = 1500
    
    def get_resized_and_scales(im):
        h, w = im.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(im, (new_w, new_h)), scale, scale
        return im, 1.0, 1.0

    tpl_resized, scale_x_t, scale_y_t = get_resized_and_scales(template)
    gray_tpl = cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2GRAY) if len(tpl_resized.shape) == 3 else tpl_resized

    sift = cv2.SIFT_create(nfeatures=5000)
    kp_t, des_t = sift.detectAndCompute(gray_tpl, None)
    
    if des_t is None:
        logger.warning("Template has no features.")
        return {"status": "fail", "aligned_image": None, "match_count": 0}

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    variants = {
        "original": image,
        "flipped": cv2.flip(image, 1)  # Horizontal flip to handle mirrored prints
    }

    best_status = "fail"
    best_img = None
    best_match_count = -1
    best_M_full = None

    for name, img_cand in variants.items():
        img_resized, scale_x_i, scale_y_i = get_resized_and_scales(img_cand)
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if len(img_resized.shape) == 3 else img_resized
        
        kp_i, des_i = sift.detectAndCompute(gray_img, None)
        
        if des_i is None:
            continue

        try:
            matches = bf.knnMatch(des_i, des_t, k=2)
        except Exception as e:
            logger.error(f"KNN match failed: {e}")
            continue

        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        match_count = len(good_matches)
        logger.debug(f"Variant '{name}': {match_count} good matches")
        
        if match_count < 15:
            continue

        src_pts = np.float32([kp_i[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M_resized, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M_resized is None:
            continue

        # Calculate actual RANSAC inliers
        inliers = int(np.sum(mask)) if mask is not None else 0
        if inliers > best_match_count:
            best_match_count = inliers
            best_status = "success"
            
            # Map transform to full-resolution image space
            S_img = np.array([
                [scale_x_i, 0, 0],
                [0, scale_y_i, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            
            S_tpl_inv = np.array([
                [1.0 / scale_x_t, 0, 0],
                [0, 1.0 / scale_y_t, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            
            best_M_full = S_tpl_inv @ M_resized @ S_img
            
            # Apply warp
            h, w = template.shape[:2]
            best_img = cv2.warpPerspective(img_cand, best_M_full, (w, h))

    if best_status == "success":
        logger.info(f"Universal align OK. Inliers: {best_match_count}")
        return {
            "status": "success",
            "aligned_image": best_img,
            "match_count": best_match_count
        }
    
    logger.warning("Universal align FAILED.")
    return {"status": "fail", "aligned_image": image, "match_count": best_match_count}
