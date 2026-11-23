"""Query normalisation: map everyday Vietnamese to the wording used in legal texts.

A small hand-written glossary (not learned). Matching terms are appended to the query, so the original wording
is kept and the legal wording is added for retrieval and clause selection.
"""

import re

GLOSSARY = {
    "vượt đèn đỏ": "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    "đèn đỏ": "hiệu lệnh của đèn tín hiệu giao thông",
    "xe máy": "xe mô tô, xe gắn máy",
    "bằng lái": "giấy phép lái xe",
    "say rượu": "trong máu hoặc hơi thở có nồng độ cồn",
    "uống rượu bia": "trong máu hoặc hơi thở có nồng độ cồn",
    "nồng độ cồn": "trong máu hoặc hơi thở có nồng độ cồn",
    "không đội mũ bảo hiểm": "không đội mũ bảo hiểm cho người đi mô tô, xe máy",
    "chạy quá tốc độ": "điều khiển xe chạy quá tốc độ quy định",
    "đi ngược chiều": "đi ngược chiều của đường một chiều",
    "nghỉ phép năm": "nghỉ hằng năm",
    "nghỉ phép": "nghỉ hằng năm",
    "đuổi việc": "đơn phương chấm dứt hợp đồng lao động",
    "nghỉ việc": "chấm dứt hợp đồng lao động",
    "lương tối thiểu": "mức lương tối thiểu vùng",
    "sổ đỏ": "giấy chứng nhận quyền sử dụng đất",
    "căn cước công dân": "thẻ căn cước",
}


def expand_query(q: str) -> str:
    low = q.lower()
    extra = []
    for k, v in GLOSSARY.items():
        if re.search(r"\b" + re.escape(k) + r"\b", low) and v.lower() not in low and v not in extra:
            extra.append(v)
    return q if not extra else f"{q} ({'; '.join(extra)})"
