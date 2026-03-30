"""
text_comparator.py — So sánh text giữa maket và ảnh chụp cho từng zone.

Thuật toán:
  1. Normalize text (lowercase, strip, gộp khoảng trắng)
  2. Fuzzy match từng dòng text maket với text chụp
  3. Phát hiện: thiếu, thừa, sai chữ
  4. Tính similarity score tổng thể
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class TextDiff:
    """Chi tiết một lỗi text."""
    diff_type: str          # "missing" | "extra" | "modified"
    expected: str           # text trên maket
    actual: str             # text trên ảnh chụp (rỗng nếu missing)
    similarity: float       # 0.0-1.0 (1.0 = giống nhau)
    detail: str = ""        # mô tả lỗi


@dataclass
class ZoneCompareResult:
    """Kết quả so sánh 1 zone."""
    zone_id: int
    zone_label: str
    status: int                        # 0=có lỗi, 1=OK
    maket_text: str                    # Full text từ maket
    chup_text: str                     # Full text từ chụp
    diffs: List[TextDiff] = field(default_factory=list)
    similarity_score: float = 1.0      # Tổng thể 0.0-1.0
    error_summary: str = ""            # Tóm tắt lỗi


def _normalize(text: str) -> str:
    """Chuẩn hóa text để so sánh: lowercase, bỏ thừa space, trim."""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    # Bỏ dấu chấm câu cuối nếu chỉ có 1
    text = text.rstrip('.,;:')
    return text


def _is_dimension_annotation(text: str) -> bool:
    """Kiểm tra text có phải annotation kích thước (52mm, 343 mm...)."""
    cleaned = text.strip().lower()
    # Pattern: số + mm/cm/inch
    if re.match(r'^\d+[\s]*(?:mm|cm|m|inch|in)$', cleaned):
        return True
    return False


def _fuzzy_match_lines(
    maket_lines: List[str],
    chup_lines: List[str],
    threshold: float = 0.6
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match từng dòng maket với dòng chụp gần nhất.

    Returns:
        matched: [(maket_idx, chup_idx, similarity)]
        unmatched_maket: [idx] — thiếu trên chụp
        unmatched_chup: [idx] — thừa trên chụp
    """
    n_m = len(maket_lines)
    n_c = len(chup_lines)

    # Tính ma trận similarity
    sim_matrix = []
    for i, ml in enumerate(maket_lines):
        row = []
        for j, cl in enumerate(chup_lines):
            sim = SequenceMatcher(None, _normalize(ml), _normalize(cl)).ratio()
            row.append(sim)
        sim_matrix.append(row)

    matched = []
    used_maket = set()
    used_chup = set()

    # Greedy match: lấy cặp similarity cao nhất trước
    pairs = []
    for i in range(n_m):
        for j in range(n_c):
            pairs.append((sim_matrix[i][j], i, j))
    pairs.sort(reverse=True)

    for sim, i, j in pairs:
        if i in used_maket or j in used_chup:
            continue
        if sim >= threshold:
            matched.append((i, j, sim))
            used_maket.add(i)
            used_chup.add(j)

    unmatched_maket = [i for i in range(n_m) if i not in used_maket]
    unmatched_chup = [j for j in range(n_c) if j not in used_chup]

    return matched, unmatched_maket, unmatched_chup


def compare_zone_texts(
    maket_text: str,
    chup_text: str,
    zone_id: int = 0,
    zone_label: str = "",
    similarity_threshold: float = 0.6,
    exact_match_threshold: float = 0.95,
) -> ZoneCompareResult:
    """
    So sánh text OCR giữa maket và ảnh chụp cho 1 zone.

    Args:
        maket_text: full text từ OCR maket zone
        chup_text: full text từ OCR chụp zone
        similarity_threshold: ngưỡng fuzzy match (< threshold = không match)
        exact_match_threshold: ngưỡng coi là "đúng" (>= threshold = OK)

    Returns:
        ZoneCompareResult
    """
    # Tách dòng, bỏ dòng trống, bỏ annotation kích thước
    maket_lines = [l.strip() for l in maket_text.split('\n') if l.strip()]
    chup_lines = [l.strip() for l in chup_text.split('\n') if l.strip()]

    # Lọc bỏ dimension annotations
    maket_lines = [l for l in maket_lines if not _is_dimension_annotation(l)]
    chup_lines = [l for l in chup_lines if not _is_dimension_annotation(l)]

    # Trường hợp đặc biệt: cả 2 đều rỗng
    if not maket_lines and not chup_lines:
        return ZoneCompareResult(
            zone_id=zone_id, zone_label=zone_label,
            status=1, maket_text="", chup_text="",
            similarity_score=1.0,
        )

    # Fuzzy match
    matched, unmatched_maket, unmatched_chup = _fuzzy_match_lines(
        maket_lines, chup_lines, similarity_threshold
    )

    diffs = []

    # 1. Matched nhưng không exact → sai chữ
    for m_idx, c_idx, sim in matched:
        if sim < exact_match_threshold:
            diffs.append(TextDiff(
                diff_type="modified",
                expected=maket_lines[m_idx],
                actual=chup_lines[c_idx],
                similarity=sim,
                detail=f"Sai chữ (giống {sim:.0%})",
            ))

    # 2. Unmatched maket → thiếu trên chụp
    for m_idx in unmatched_maket:
        diffs.append(TextDiff(
            diff_type="missing",
            expected=maket_lines[m_idx],
            actual="",
            similarity=0.0,
            detail="Thiếu text (có trên maket, không có trên bản in)",
        ))

    # 3. Unmatched chụp → thừa trên chụp
    for c_idx in unmatched_chup:
        diffs.append(TextDiff(
            diff_type="extra",
            expected="",
            actual=chup_lines[c_idx],
            similarity=0.0,
            detail="Thừa text (có trên bản in, không có trên maket)",
        ))

    # Tính similarity tổng thể
    total_lines = max(len(maket_lines), len(chup_lines), 1)
    exact_matches = sum(1 for _, _, s in matched if s >= exact_match_threshold)
    overall_sim = exact_matches / total_lines

    # Status
    has_error = len(diffs) > 0
    status = 0 if has_error else 1

    # Error summary
    error_parts = []
    n_missing = sum(1 for d in diffs if d.diff_type == "missing")
    n_extra = sum(1 for d in diffs if d.diff_type == "extra")
    n_modified = sum(1 for d in diffs if d.diff_type == "modified")
    if n_missing:
        error_parts.append(f"thiếu {n_missing} dòng")
    if n_extra:
        error_parts.append(f"thừa {n_extra} dòng")
    if n_modified:
        error_parts.append(f"sai {n_modified} dòng")
    error_summary = ", ".join(error_parts) if error_parts else ""

    return ZoneCompareResult(
        zone_id=zone_id,
        zone_label=zone_label,
        status=status,
        maket_text="\n".join(maket_lines),
        chup_text="\n".join(chup_lines),
        diffs=diffs,
        similarity_score=overall_sim,
        error_summary=error_summary,
    )


def apply_maket_zones_to_image(
    image,
    maket_zones: List[Dict],
) -> List[Dict]:
    """
    Áp dụng tọa độ zone của maket lên ảnh chụp (đã resize cùng kích thước).
    Trả về list zone dicts giống format của zone_splitter nhưng crop từ ảnh chụp.

    QUAN TRỌNG: ảnh chụp PHẢI đã được resize cùng kích thước maket trước.
    """
    h_img, w_img = image.shape[:2]
    result_zones = []

    for z in maket_zones:
        x1, y1, x2, y2 = z["bbox"]
        # Clamp to image bounds
        x1 = max(0, min(x1, w_img))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img))
        y2 = max(0, min(y2, h_img))

        if x2 <= x1 or y2 <= y1:
            continue

        zone_img = image[y1:y2, x1:x2].copy()
        result_zones.append({
            "zone_id": z["zone_id"],
            "label": z["label"],
            "row": z["row"],
            "col": z["col"],
            "bbox": (x1, y1, x2, y2),
            "image": zone_img,
        })

    return result_zones


def compare_all_zones(
    maket_zones: List[Dict],
    chup_zones: List[Dict],
    maket_texts: Dict[int, str],
    chup_texts: Dict[int, str],
) -> Dict[str, Any]:
    """
    So sánh tất cả zones, trả về kết quả tổng hợp.

    Args:
        maket_zones: list zone dicts từ zone_splitter
        chup_zones: list zone dicts (cùng grid, crop từ ảnh chụp)
        maket_texts: {zone_id: full_text} từ OCR
        chup_texts: {zone_id: full_text} từ OCR

    Returns:
        {
            "status": 0/1,
            "total_zones": int,
            "error_zones": int,
            "zone_results": [ZoneCompareResult],
            "summary": str,
        }
    """
    zone_results = []

    # Map zones by zone_id for matching (cùng grid → cùng zone_id)
    maket_by_id = {z["zone_id"]: z for z in maket_zones}
    chup_by_id = {z["zone_id"]: z for z in chup_zones}

    # So sánh từng zone tương ứng
    for zone_id in sorted(maket_by_id.keys()):
        mz = maket_by_id[zone_id]
        cz = chup_by_id.get(zone_id)

        m_text = maket_texts.get(zone_id, "")
        c_text = chup_texts.get(zone_id, "") if cz else ""

        # Bỏ qua zone mà cả 2 đều rỗng
        if not m_text.strip() and not c_text.strip():
            zone_results.append(ZoneCompareResult(
                zone_id=zone_id,
                zone_label=mz["label"],
                status=1,
                maket_text="",
                chup_text="",
                similarity_score=1.0,
            ))
            continue

        result = compare_zone_texts(
            maket_text=m_text,
            chup_text=c_text,
            zone_id=zone_id,
            zone_label=mz["label"],
        )
        zone_results.append(result)

    # Tổng hợp — chỉ tính zone có text
    zones_with_text = [r for r in zone_results if r.maket_text.strip() or r.chup_text.strip()]
    error_zones = [r for r in zones_with_text if r.status == 0]
    overall_status = 0 if error_zones else 1

    summary_parts = []
    for ez in error_zones:
        summary_parts.append(f"[{ez.zone_label}] {ez.error_summary}")
    summary = "; ".join(summary_parts) if summary_parts else "Tất cả zones đạt"

    return {
        "status": overall_status,
        "total_zones": len(zones_with_text),
        "error_zones": len(error_zones),
        "zone_results": zone_results,
        "summary": summary,
    }

