"""Grounded answer generation with a self-hosted Qwen3-4B (vLLM, OpenAI-compatible API)."""

from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp pháp luật Việt Nam.
Quy tắc:
1. Chỉ trả lời dựa trên các trích đoạn văn bản pháp luật được cung cấp. Không dùng kiến thức bên ngoài.
2. Mỗi ý quan trọng phải có trích dẫn dạng [số] tương ứng với trích đoạn, kèm số hiệu văn bản và điều, khoản.
3. Ưu tiên các trích đoạn đang CÒN HIỆU LỰC tại thời điểm hỏi. Nếu một quy định đã bị sửa đổi, thay thế hoặc hết hiệu lực,
   thêm mục "⚠️ Thay đổi pháp luật" nêu rõ: quy định cũ (văn bản, nội dung chính), quy định mới (văn bản, nội dung chính)
   và thời điểm thay đổi.
4. Nếu các trích đoạn không đủ căn cứ để trả lời, nói rõ "Không tìm thấy đủ căn cứ pháp lý" và không suy đoán.
5. Trả lời ngắn gọn, rõ ràng, bằng tiếng Việt."""


def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and x != x) else str(x)


def _fmt_date(x) -> str:
    return pd.Timestamp(x).strftime("%d/%m/%Y") if pd.notna(x) else "không rõ"


def build_context(results: pd.DataFrame, lg, t: pd.Timestamp, time_aware: bool, n_sources: int = 5,
                  max_chars: int = 1800) -> tuple[str, list[dict]]:
    """Pick sources and format them. Time-aware mode prefers valid articles and adds successor articles."""
    rows = results.to_dict("records")
    if time_aware:
        valid = [r for r in rows if not r["stale"]][:n_sources]
        chosen = list(valid)
        for r in rows[:10]:
            if r["stale"] and _s(r.get("successor_aid")) and len(chosen) < n_sources + 3:
                chosen.append(r)
                succ_rows = lg.rows_of_aid.get(r["successor_aid"], [])
                if succ_rows and not any(c.get("aid") == r["successor_aid"] for c in chosen):
                    s = lg.corpus.iloc[succ_rows[0]].to_dict()
                    s.update({"stale": False, "row": succ_rows[0], "is_successor_of": r.get("so_ky_hieu")})
                    chosen.append(s)
    else:
        chosen = rows[:n_sources]

    blocks, sources = [], []
    for k, r in enumerate(chosen, start=1):
        doc = _s(r.get("so_ky_hieu")) or "văn bản chưa xác định"
        title = _s(r.get("title"))[:160]
        head = f"[{k}] {doc} — {title}"
        if time_aware:
            if r.get("stale"):
                note = f"ĐÃ HẾT HIỆU LỰC / BỊ THAY THẾ tại {_fmt_date(t)}"
                if _s(r.get("successor_doc")):
                    note += f" (được thay thế bởi văn bản {lg.docs.so_ky_hieu.get(r['successor_doc'], r['successor_doc'])})"
            else:
                note = f"còn hiệu lực tại {_fmt_date(t)} (hiệu lực từ {_fmt_date(r.get('valid_from'))})"
            head += f"\nTình trạng: {note}"
        text = str(r.get("text", ""))[:max_chars]
        blocks.append(f"{head}\n{text}")
        sources.append({"k": k, "doc": doc, "title": title, "aid": _s(r.get("aid")), "stale": bool(r.get("stale")),
                        "row": int(r["row"]) if _s(r.get("row")) else None})
    return "\n\n".join(blocks), sources


def answer(client, model: str, question: str, context: str, t: pd.Timestamp, thinking: bool = False,
           max_tokens: int = 1024) -> str:
    user = (f"Thời điểm hỏi: {_fmt_date(t)}\n\nCác trích đoạn văn bản pháp luật:\n{context}\n\n"
            f"Câu hỏi: {question}")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        temperature=0.3 if not thinking else 0.6,
        top_p=0.8 if not thinking else 0.95,
        max_tokens=max_tokens if not thinking else 4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}, "top_k": 20},
    )
    text = resp.choices[0].message.content or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
