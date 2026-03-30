import os
import sys
import asyncio

try:
    from apps.web_api.routers.kiem_tra_in.schemas import KiemTraInRequest
    from apps.web_api.routers.kiem_tra_in.router import kiem_tra_in
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

async def main():
    req = KiemTraInRequest(
        chup_image=r"C:\Users\Dung.NT\Pictures\A2.JPG",
        maket_image=r"D:\onedriver\OneDrive\Bao_bi_carton\Github\DK2IMS\public\upload\markets\2SP218062-1.jpg",
        loai=1,
        phoi_dai=1000,
        phoi_rong=800,
        chieu_dai=500,
        chieu_rong=300,
        chieu_cao=200,
        mau_nen="trang",
        mau_giay="trang",
        ma_san_pham="2SP218062-1",
        ten_maket="Test Maket"
    )
    
    sys.stdout.reconfigure(encoding='utf-8')
    print("Environment ready, calling kiem_tra_in()...")
    try:
        res = await kiem_tra_in(req)
        print("\n--- KẾT QUẢ ---")
        print(res.model_dump_json(indent=2))
        print("---------------")
        
        # Save output to a file
        with open("test_kiemtra_output.json", "w", encoding="utf-8") as f:
            f.write(res.model_dump_json(indent=2))
        print(f"Saved output to test_kiemtra_output.json")
    except Exception as e:
        print(f"Lỗi khi thực thi kiem_tra_in: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
