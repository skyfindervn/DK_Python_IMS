import cv2
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import importlib.util

MODULE_DIR = os.path.join(os.path.dirname(__file__), "..", "web_api", "routers", "kiem_tra_in")
ocr_engine = importlib.util.spec_from_file_location("ocr_engine", os.path.join(MODULE_DIR, "ocr_engine.py"))
ocr_mod = importlib.util.module_from_spec(ocr_engine)
ocr_engine.loader.exec_module(ocr_mod)

def main():
    MAKET = r"D:\onedriver\OneDrive\Bao_bi_carton\Github\DK2IMS\public\upload\markets\2SP218062-1.jpg"
    img_maket = cv2.imread(MAKET)
    if img_maket is None:
        print("Không load được ảnh!")
        return
        
    print("Shape ảnh gốc:", img_maket.shape)
    
    # R1C3 = (4168, 0, 6532, 829)
    # Nhưng ảnh MAKET ở đây là 9402x4722. Crop bounding box trong log là sau khi đã xử lý crop_maket_by_border.
    # Tốt nhất là ta crop theo giống vậy bằng tay:
    from pathlib import Path
    cr_maket_path = Path(os.path.join(os.path.dirname(__file__), "test_maket_cropped.jpg"))
    if not cr_maket_path.exists():
        print("Không tìm thấy test_maket_cropped.jpg")
        return
    img = cv2.imread(str(cr_maket_path))
    
    import time
    log_f = open("crash_debug_log.txt", "w", encoding="utf-8")
    def p(text):
        print(text)
        log_f.write(text + "\n")
        log_f.flush()

    p(f"Shape ảnh crop: {img.shape}")
    zones = [
        ("R1C1", 0, 0, 2514, 829),
        ("R1C2", 2514, 0, 4168, 829),
        ("R1C3", 4168, 0, 6532, 829),
        ("R1C4", 6532, 0, 7469, 829),
        ("R1C5", 7469, 0, 8171, 829),
    ]
    
    for label, x1, y1, x2, y2 in zones:
        zone_img = img[y1:y2, x1:x2].copy()
        p(f"\n=================\nChạy OCR cho zone {label}, shape: {zone_img.shape}")
        
        try:
            time.sleep(0.5)
            blocks = ocr_mod.ocr_zone(zone_img)
            text = ocr_mod.blocks_to_text(blocks)
            p(f"Kết quả {label}: {repr(text)}")
        except Exception as e:
            p(f"Exception caught on {label}: {e}")
            
    p("\nDone!")
    log_f.close()

if __name__ == "__main__":
    main()
