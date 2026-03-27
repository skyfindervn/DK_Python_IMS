"""
llm_client.py – Gọi LLM API để so sánh từng block ảnh chụp vs maket.

Hỗ trợ format OpenAI-compatible (Gemini, OpenAI, Azure, Groq...).
Endpoint và key lấy từ DB qua Laravel (cau_hinh_chung CH-002).
"""

import base64
import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Prompt mặc định nếu DB không có
DEFAULT_PROMPT = """
Bạn là chuyên gia kiểm tra chất lượng in ấn bao bì carton.

Tôi cung cấp cho bạn 2 ảnh:
- Ảnh 1 (MAKET): Bản mẫu chuẩn
- Ảnh 2 (THỰC TẾ): Ảnh chụp sản phẩm vừa in ra

Hãy so sánh chi tiết và trả về JSON với format:
{
  "status": 0 hoặc 1,           // 0=có lỗi, 1=ok
  "noi_dung_loi": "...",        // Mô tả lỗi nội dung (thiếu chữ, sai chữ, sai chính tả...)
  "vi_tri_loi": "..."           // Mô tả lỗi vị trí (lệch, méo, không đúng tỉ lệ...)
}

Nếu không có lỗi, trả về: {"status": 1, "noi_dung_loi": null, "vi_tri_loi": null}

QUAN TRỌNG:
- Chỉ trả về JSON thuần túy, không giải thích thêm.
- Kiểm tra kỹ: chữ viết hoa/thường, dấu câu, tên riêng, logo, khoảng cách.
- Ưu tiên phát hiện: thiếu chữ, sai chính tả, sai vị trí logo/text.
"""


def call_llm(
    endpoint: str,
    api_key: str,
    prompt_template: str,
    img_chup_b64: str,
    img_maket_b64: str,
    block_label: str = "Block",
) -> dict:
    """
    Gọi LLM API với 2 ảnh base64 (maket + thực tế) và prompt.
    Trả về dict: {status, noi_dung_loi, vi_tri_loi, raw}
    """
    prompt = prompt_template.strip() if prompt_template and prompt_template.strip() else DEFAULT_PROMPT
    
    # Chuẩn hóa endpoint (tránh lỗi thiếu protocol từ Laravel database)
    if not endpoint or not str(endpoint).strip():
        return _error_result("LLM Endpoint bị bỏ trống trong cấu hình. (Kiểm tra lại DB: cau_hinh_chung CH-002)")
        
    endpoint = str(endpoint).strip()
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = "https://" + endpoint
        logger.info(f"Auto-prepended https:// to endpoint: {endpoint}")

    # Xây dựng payload theo OpenAI vision format (compatible với Gemini, Azure...)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}\n\n[Đang kiểm tra: {block_label}]"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_maket_b64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_chup_b64}",
                        "detail": "high"
                    }
                },
            ],
        }
    ]

    payload = {
        "model": "gpt-4o",          # Sẽ bị override nếu endpoint là Gemini
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.1,         # Low temp để kết quả nhất quán
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    raw_text = ""
    try:
        with httpx.Client(timeout=60.0) as client:
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


def _parse_llm_output(text: str) -> dict:
    """Parse JSON từ LLM response. Trả về dict chuẩn."""
    text = text.strip()
    # Tìm JSON trong response (đôi khi LLM wrap trong markdown ```json ... ```)
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end] if start != -1 else text

    try:
        obj = json.loads(text)
        return {
            "status": int(obj.get("status", 0)),
            "noi_dung_loi": obj.get("noi_dung_loi") or None,
            "vi_tri_loi": obj.get("vi_tri_loi") or None,
        }
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"Không parse được LLM JSON: {text[:100]}")
        # Nếu không parse được → coi như có lỗi
        return {
            "status": 0,
            "noi_dung_loi": f"Không parse được kết quả LLM: {text[:200]}",
            "vi_tri_loi": None,
        }


def _error_result(msg: str, raw: str = "") -> dict:
    return {
        "status": 0,
        "noi_dung_loi": f"[Lỗi gọi LLM] {msg}",
        "vi_tri_loi": None,
        "raw": raw,
    }
