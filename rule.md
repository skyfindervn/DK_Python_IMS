# Luật Kiến Trúc Monorepo

> Tài liệu này là **bắt buộc** đối với mọi contributor và mọi AI agent làm việc trong Workspace này.  
> Mọi Pull Request / commit **vi phạm** bất kỳ luật nào dưới đây đều phải bị từ chối.

---

## Luật 1 – Cô Lập Tuyệt Đối giữa các App

- Các thư mục bên trong `apps/` **không được import lẫn nhau**.
- Ví dụ: `apps/web_api/` **không được** `import` bất kỳ module nào từ `apps/data_crawler/` và ngược lại.

```
# ❌ Vi phạm
from apps.data_crawler.some_module import something

# ✅ Đúng – dùng shared
from shared.some_module import something
```

---

## Luật 2 – Code Dùng Chung Phải Đặt ở `shared/`

- Mọi logic được dùng bởi **từ 2 app trở lên** (database, utils, models, config…) phải đặt trong `shared/src/shared/`.
- Không được duplicate logic giống nhau ở nhiều app.

```
shared/
└── src/shared/
    ├── database.py   ← kết nối DB
    ├── utils.py      ← helper functions
    └── config.py     ← cấu hình chung
```

---

## Luật 3 – Quản Lý Package bắt buộc dùng `uv`

- **TUYỆT ĐỐI KHÔNG** dùng `pip install`.
- Khi cần cài thư viện mới cho một app cụ thể:

```powershell
uv add <tên_package> --project apps/<tên_app>
```

- Khi cần cài thư viện cho `shared`:

```powershell
uv add <tên_package> --project shared
```

- Sau mỗi lần thêm package, file `uv.lock` sẽ tự cập nhật và **phải được commit** cùng với `pyproject.toml`.

---

## Luật 4 – Một Entry Point duy nhất

- Mỗi app chỉ có **đúng 1** file khởi chạy: `main.py` nằm ngay ở gốc thư mục app.
- Không được tạo thêm `run.py`, `app.py`, `server.py`… với vai trò tương đương.

```
apps/
├── web_api/
│   └── main.py       ← ✅ entry point duy nhất
└── data_crawler/
    └── main.py       ← ✅ entry point duy nhất
```

---

## Tóm Tắt Nhanh

| # | Luật | ✅ Đúng | ❌ Sai |
|---|------|---------|--------|
| 1 | Cô lập app | `from shared.x import y` | `from apps.other.x import y` |
| 2 | Code chung ở shared | `shared/src/shared/utils.py` | Duplicate ở nhiều app |
| 3 | Quản lý package | `uv add requests --project apps/web_api` | `pip install requests` |
| 4 | Entry point | `apps/web_api/main.py` | `apps/web_api/run.py` |
