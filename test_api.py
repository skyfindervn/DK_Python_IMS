import sys
import json

try:
    import requests
except ImportError:
    print("Requests not installed, attempting to use urllib")
    import urllib.request
    import urllib.parse
    requests = None

url = "http://localhost:2342/api/kiem-tra-in"
payload = {
    "chup_image": r"C:\Users\Dung.NT\Pictures\A2.JPG",
    "maket_image": r"D:\onedriver\OneDrive\Bao_bi_carton\Github\DK2IMS\public\upload\markets\2SP218062-1.jpg",
    "loai": 1,
    "phoi_dai": 1000,
    "phoi_rong": 800,
    "chieu_dai": 500,
    "chieu_rong": 300,
    "chieu_cao": 200,
    "mau_nen": "trang",
    "mau_giay": "trang",
    "ma_san_pham": "2SP218062-1",
    "ten_maket": "2SP218062-1"
}

sys.stdout.reconfigure(encoding='utf-8')
print("Sending request to FastAPI at", url, "...")

if requests:
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"Status Code: {response.status_code}")
        
        try:
            data = response.json()
            # Save to file
            with open("test_api_result.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("Summary:")
            if "message" in data:
                print("Message:", data["message"])
            if "result" in data:
                print("Result Status:", data["result"].get("status"))
                print("Error Note:", data["result"].get("noi_dung_loi"))
            
            print("Successfully saved full JSON to test_api_result.json")
        except Exception as e:
            print("Failed to parse JSON:", e)
            print(response.text)
    except Exception as e:
        print(f"Request Error: {e}")
        print("Maybe the FastAPI server is not running on port 2342? Run start_service.bat to start it.")
else:
    # Fallback to urllib
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=120) as response:
            result = response.read().decode('utf-8')
            print(f"Status Code: {response.getcode()}")
            
            try:
                data_json = json.loads(result)
                with open("test_api_result.json", "w", encoding="utf-8") as f:
                    json.dump(data_json, f, indent=2, ensure_ascii=False)
                
                print("Summary:")
                if "message" in data_json:
                    print("Message:", data_json["message"])
                if "result" in data_json:
                    print("Result Status:", data_json["result"].get("status"))
                    print("Error Note:", data_json["result"].get("noi_dung_loi"))
                    
                print("Successfully saved full JSON to test_api_result.json")
            except Exception as e:
                print("Failed to parse JSON:", e)
                print(result)
    except Exception as e:
        print(f"Request Error: {e}")
        print("Maybe the FastAPI server is not running on port 2342? Run start_service.bat to start it.")
