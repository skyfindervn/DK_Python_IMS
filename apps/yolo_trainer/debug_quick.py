"""Quick test for maket #2 to debug OCR crash."""
import sys, os, io, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import importlib.util, cv2, numpy as np
from pathlib import Path

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

MODULE_DIR = r'd:\DK_Python_IMS\DK_Python_IMS\apps\web_api\routers\kiem_tra_in'
ocr_engine = _import_module('ocr_engine', os.path.join(MODULE_DIR, 'ocr_engine.py'))
zone_splitter = _import_module('zone_splitter', os.path.join(MODULE_DIR, 'zone_splitter.py'))

f = Path(r'd:\onedriver\OneDrive\Bao_bi_carton\Github\DK2IMS\public\upload\markets\2SP217118-1.jpg')
img = cv2.imread(str(f))
print(f'Loaded: {img.shape}')

zones = zone_splitter.split_into_zones(img)
print(f'Zones: {len(zones)}')
for z in zones[:2]:
    try:
        print(f'OCR zone {z["label"]} ...')
        blocks = ocr_engine.ocr_zone(z['image'])
        text = ocr_engine.blocks_to_text(blocks)
        if text:
            print(f'  Result: {text[:100]}')
        else:
            print('  (empty)')
        print('OK')
    except Exception as e:
        traceback.print_exc()
        print(f'ERROR: {e}')

print("ALL DONE")
