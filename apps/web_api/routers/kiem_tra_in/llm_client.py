import base64
import json
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# File prompt mặc định — nằm cùng thư mục với llm_client.py
_PROMPT_FILE = Path(__file__).parent / "prompt.txt"

_PROMPT_FALLBACK = (
    "So sánh ảnh MAKET (ảnh 1) và ảnh THỰC TẾ (ảnh 2). "
    "Trả về JSON: {\"status\":0/1, \"noi_dung_loi\":\"...\", \"vi_tri_loi\":\"...\"} "
    "Chỉ trả JSON thuần, không giải thích."
)


def _load_default_prompt() -> str:
    """Đọc prompt từ file prompt.txt. Reload mỗi lần gọi (không cần restart service)."""
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning(f"Không đọc được {_PROMPT_FILE}: {e} — dùng prompt fallback")
        return _PROMPT_FALLBACK


def call_llm(
    endpoint: str,
    api_key: str,
    prompt_template: str,
    img_chup_b64: str,
    img_maket_b64: str,
    block_label: str = "Block",
    extra_context: str = "",
) -> dict:
    """
    Gọi LLM API với 2 ảnh base64 (maket + thực tế) và prompt.
    extra_context: kết quả phân tích OpenCV để gắn vào prompt.
    Trả về dict: {status, noi_dung_loi, vi_tri_loi, raw}
    """
    prompt = prompt_template.strip() if prompt_template and prompt_template.strip() else _load_default_prompt()
    # Gắn kết quả phân tích OpenCV vào prompt nếu có
    full_prompt = prompt
    if extra_context:
        full_prompt = prompt + "\n\n" + extra_context

    # Chuẩn hóa endpoint (tránh lỗi thiếu protocol từ Laravel database)
    if not endpoint or not str(endpoint).strip():
        return _error_result("LLM Endpoint bị bỏ trống trong cấu hình. (Kiểm tra lại DB: cau_hinh_chung CH-002)")
        
    endpoint = str(endpoint).strip()
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = "https://" + endpoint
        logger.info(f"Auto-prepended https:// to endpoint: {endpoint}")

    # ── Auto-detect model name từ server (/v1/models) ────────────────────────
    model_name = "gpt-4o"   # default fallback (OpenAI)
    try:
        base_url = endpoint.rsplit("/chat/completions", 1)[0]  # bỏ đuôi /chat/completions
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code == 200:
                models_data = r.json().get("data", [])
                if models_data:
                    model_name = models_data[0].get("id", model_name)
                    logger.info(f"Auto-detected model: {model_name}")
    except Exception as _e:
        logger.warning(f"Không lấy được model list, dùng mặc định '{model_name}': {_e}")

    # Xây dựng payload theo OpenAI vision format
    # - Nếu img_maket_b64 rỗng: chế độ 1 ảnh (ghép sẵn trong img_chup_b64)
    # - Nếu img_maket_b64 có: chế độ 2 ảnh riêng (cho model hỗ trợ multi-image)
    content = [
        {
            "type": "text",
            "text": f"{full_prompt}\n\n[Đang kiểm tra: {block_label}]"
        },
    ]

    if img_maket_b64:
        # Chế độ 2 ảnh: maket trước, thực tế sau
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_maket_b64}", "detail": "high"}
        })

    # Ảnh thực tế (hoặc ảnh ghép nếu chế độ 1 ảnh)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_chup_b64}", "detail": "high"}
    })

    messages = [{"role": "user", "content": content}]

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 2048,   # InternVL2-6B đã cấu hình 8192 tổng → chừa ~6000 cho ảnh nét + prompt
        "temperature": 0.1,     # Low temp để kết quả nhất quán
        # response_format bị bỏ — nhiều local server (vLLM, Ollama) không hỗ trợ
        # Thay vào đó prompt yêu cầu trả JSON thuần (đã có trong DEFAULT_PROMPT)
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    raw_text = ""
    try:
        with httpx.Client(timeout=120.0) as client:   # 120s cho local model xử lý ảnh lớn
            response = client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        # Parse response (OpenAI format)
        raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Nếu endpoint là Gemini direct
        if not raw_text and "candidates" in data:
            raw_text = data["candidates"][0]["content"]["parts"][0].get("text", "")

        result = _parse_llm_output(raw_text)
        result["raw"] = raw_text
        return result

    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API HTTP error: {e.response.status_code} — {e.response.text[:200]}")
        return _error_result(f"LLM HTTP {e.response.status_code}", raw_text)
    except httpx.TimeoutException:
        logger.error("LLM API timeout")
        return _error_result("LLM timeout", raw_text)
    except Exception as e:
        logger.error(f"LLM exception: {e}")
        return _error_result(str(e), raw_text)


def _extract_json(text: str) -> str:
    """Trích xuất JSON hợp lệ từ text (bỏ markdown, text thừa trước/sau)."""
    text = text.strip()
    # Bỏ markdown code block
    if "```" in text:
        for fence in ["```json", "```"]:
            if fence in text:
                text = text.split(fence, 1)[-1]
                text = text.rsplit("```", 1)[0]
                break
        text = text.strip()
    # Tìm JSON object lớn nhất (từ { đầu tiên đến } cuối cùng)
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return text


def _parse_llm_output(text: str) -> dict:
    """
    Parse JSON từ LLM response. Hỗ trợ 2 schema:
    - Schema mới (prompt.txt): {"blocks": [{"block_id":..., "block_final_result":"ok"/"not oki", "evaluation":{...}}]}
    - Schema cũ (fallback):    {"status": 0/1, "noi_dung_loi": "...", "vi_tri_loi": "..."}
    """
    text = _extract_json(text)

    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Không parse được LLM JSON ({e}): {text[:500]}")
        return {
            "status": 0,
            "noi_dung_loi": f"Không parse được kết quả LLM (JSON bị truncate?): {text[:300]}",
            "vi_tri_loi": None,
        }

    # ── Schema mới: {"blocks": [...]} ────────────────────────────────────────
    if "blocks" in obj and isinstance(obj["blocks"], list):
        blocks = obj["blocks"]
        failed_blocks = []
        error_contents = []
        error_positions = []

        for blk in blocks:
            result = blk.get("block_final_result", "ok")
            is_fail = (str(result).lower().strip() == "not oki")
            if is_fail:
                bid  = blk.get("block_id", "?")
                bdesc = blk.get("block_description", "")
                failed_blocks.append(f"{bid}({bdesc})")

                # Thu thập chi tiết lỗi từ evaluation
                ev = blk.get("evaluation", {})
                loi_noidung = []
                loi_vitri   = []
                for key, val in ev.items():
                    if not isinstance(val, dict):
                        continue
                    if val.get("status", "ok") != "ok":
                        alert = val.get("alert", "")
                        alert_str = f" [{alert}]" if alert else ""
                        if "vi_tri" in key or "vi tri" in key:
                            loi_vitri.append(key.replace("tieu_chi_4_", "").replace("_", " ") + alert_str)
                        else:
                            loi_noidung.append(key.replace("tieu_chi_1_", "").replace("tieu_chi_2_", "").replace("tieu_chi_3_", "").replace("_", " ") + alert_str)

                if loi_noidung:
                    error_contents.append(f"[{bid}] " + ", ".join(loi_noidung))
                if loi_vitri:
                    error_positions.append(f"[{bid}] " + ", ".join(loi_vitri))

        overall_ok = len(failed_blocks) == 0
        return {
            "status": 1 if overall_ok else 0,
            "noi_dung_loi": "; ".join(error_contents) if error_contents else None,
            "vi_tri_loi":   "; ".join(error_positions) if error_positions else None,
        }

    # ── Schema cũ: {"status": 0/1, "noi_dung_loi": "...", "vi_tri_loi": "..."} ─
    raw_status = obj.get("status", 0)
    # Chuẩn hóa "ok"/"not oki" → 1/0
    if isinstance(raw_status, str):
        raw_status = 1 if raw_status.lower().strip() == "ok" else 0
    return {
        "status": int(raw_status),
        "noi_dung_loi": obj.get("noi_dung_loi") or None,
        "vi_tri_loi":   obj.get("vi_tri_loi")   or None,
    }


def _error_result(msg: str, raw: str = "") -> dict:
    return {
        "status": 0,
        "noi_dung_loi": f"[Lỗi gọi LLM] {msg}",
        "vi_tri_loi": None,
        "raw": raw,
    }
