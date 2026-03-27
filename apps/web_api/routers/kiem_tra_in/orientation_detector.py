import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_orientation(image: np.ndarray, template: np.ndarray) -> dict:
    """
    Detects orientation using ORB feature matching.
    """
    orb = cv2.ORB_create(nfeatures=1000)
    
    h_t, w_t = template.shape[:2]
    if max(h_t, w_t) > 800:
        scale_t = 800 / max(h_t, w_t)
        tmp = cv2.resize(template, (int(w_t * scale_t), int(h_t * scale_t)))
    else:
        tmp = template.copy()
        
    gray_t = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY) if len(tmp.shape) == 3 else tmp
    kp_t, des_t = orb.detectAndCompute(gray_t, None)
    
    if des_t is None:
        logger.warning("Orientation: No features found in template.")
        return {"orientation": "original", "score": 0, "image": image}
        
    candidates = {
        "original": image,
        "rot180": cv2.rotate(image, cv2.ROTATE_180),
        "flip_h": cv2.flip(image, 1),
        "flip_v": cv2.flip(image, 0)
    }
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    best_name = "original"
    best_score = -1
    best_img = image
    
    for name, cand in candidates.items():
        h_c, w_c = cand.shape[:2]
        if max(h_c, w_c) > 800:
            scale_c = 800 / max(h_c, w_c)
            cand_resized = cv2.resize(cand, (int(w_c * scale_c), int(h_c * scale_c)))
        else:
            cand_resized = cand.copy()
            
        gray_c = cv2.cvtColor(cand_resized, cv2.COLOR_BGR2GRAY) if len(cand_resized.shape) == 3 else cand_resized
        kp_c, des_c = orb.detectAndCompute(gray_c, None)
        
        if des_c is None or len(des_c) < 10:
            continue
            
        try:
            matches = bf.knnMatch(des_c, des_t, k=2)
        except Exception:
            continue
            
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Để chắc chắn hướng nào đúng, dùng Homography tính inliers thực sự
        if len(good_matches) >= 4:
            src_pts = np.float32([kp_c[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            score = int(np.sum(mask)) if mask is not None else 0
        else:
            score = 0
            
        logger.debug(f"Orientation [{name}]: {len(good_matches)} good matches, {score} inliers")
        
        if score > best_score:
            best_score = score
            best_name = name
            best_img = cand


    logger.info(f"Orientation detect -> best='{best_name}' score={best_score}")
    return {"orientation": best_name, "score": best_score, "image": best_img}
