"""Grounded answer generation with a self-hosted Qwen3-4B (vLLM, OpenAI-compatible API)."""

from __future__ import annotations

import pandas as pd

from lexgraph.query import expand_query

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
                  max_chars: int = 1800, query: str | None = None) -> tuple[str, list[dict]]:
    """Pick sources and format them. Time-aware mode prefers valid articles and adds successor articles."""
    rows = results.to_dict("records")
    from lexgraph.query import expand_query
    q_exp = expand_query(query) if query else None
    q_vec = lg.encode_queries([q_exp])[0] if query else None
    if time_aware:
        valid = [r for r in rows if not r["stale"]][:n_sources]
        chosen = list(valid)
        chosen_aids = {_s(r.get("aid")) for r in chosen}
        top_ce = max([r["ce_score"] for r in valid], default=0.0)
        n_old = 0
        for r in rows:
            if n_old >= 3 or not (r["stale"] and _s(r.get("successor_aid"))):
                continue
            # an old version is relevant if its successor is already a source, or it scores close to the best source
            if r["successor_aid"] in chosen_aids or r["ce_score"] >= top_ce - 1.5:
                chosen.append(r)
                n_old += 1
                if r["successor_aid"] not in chosen_aids:
                    succ_rows = lg.rows_of_aid.get(r["successor_aid"], [])
                    if succ_rows:
                        s = lg.corpus.iloc[succ_rows[0]].to_dict()
                        s.update({"stale": False, "row": succ_rows[0]})
                        chosen.append(s)
                        chosen_aids.add(r["successor_aid"])
    else:
        chosen = rows[:n_sources]

    seen, uniq = set(), []
    for r in chosen:  # one block per article or chunk (the QA corpus repeats some articles)
        key = _s(r.get("chunk_id")) or _s(r.get("aid")) or f"row{r.get('row')}"
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    chosen = uniq

    blocks, sources = [], []
    for k, r in enumerate(chosen, start=1):
        doc = _s(r.get("so_ky_hieu")) or "văn bản chưa xác định"
        title = _s(r.get("title"))[:160]
        aid = _s(r.get("aid"))
        art = f", Điều {aid.split('#')[-1].split('~')[0]}" if "#" in aid else ""
        head = f"[{k}] {doc}{art} — {title}"
        if time_aware:
            if r.get("stale"):
                note = f"ĐÃ HẾT HIỆU LỰC / BỊ THAY THẾ tại {_fmt_date(t)}"
                if _s(r.get("successor_doc")):
                    note += f" (được thay thế bởi văn bản {lg.docs.so_ky_hieu.get(r['successor_doc'], r['successor_doc'])})"
            else:
                note = f"còn hiệu lực tại {_fmt_date(t)} (hiệu lực từ {_fmt_date(r.get('valid_from'))})"
            head += f"\nTình trạng: {note}"
        text = str(r.get("text", ""))
        if q_vec is not None and _s(r.get("source")) == "qa":
            text = lg.relevant_text(q_vec, aid, text, per_chunk_chars=max_chars, query_text=q_exp)[: 2 * max_chars + 10]
        else:
            text = text[:max_chars]
        versions = getattr(lg, "versions", None)
        if time_aware and versions is not None and not r.get("stale") and aid and q_vec is not None:
            am = versions.amendments(aid, t)
            am = am[am.new_text.notna()]
            if len(am):  # only the amendment most related to the question
                e = lg.bi.encode(am.new_text.str[:1500].tolist(), normalize_embeddings=True, convert_to_tensor=True)
                sims = (e @ q_vec).float().cpu().numpy()
                am = am.assign(_sim=sims)
                am = am[am._sim >= 0.55].nlargest(1, "_sim")
            for a in am.itertuples(index=False):
                text += (f"\n[Sửa đổi bởi {a.amend_num}, hiệu lực từ {_fmt_date(a.amend_from)}: {a.op} {a.spec}]\n"
                         f"{str(a.new_text)[:700]}")
        blocks.append(f"{head}\n{text}")
        sources.append({"k": k, "doc": doc, "title": title, "aid": _s(r.get("aid")), "stale": bool(r.get("stale")),
                        "row": int(r["row"]) if _s(r.get("row")) else None})
    return "\n\n".join(blocks), sources


def answer(client, model: str, question: str, context: str, t: pd.Timestamp, thinking: bool = False,
           max_tokens: int = 1024) -> str:
    user = (f"Thời điểm hỏi: {_fmt_date(t)}\n\nCác trích đoạn văn bản pháp luật:\n{context}\n\n"
            f"Câu hỏi: {expand_query(question)}")
    if "ĐÃ HẾT HIỆU LỰC / BỊ THAY THẾ" in context:
        user += ("\n\nLưu ý: có trích đoạn đã hết hiệu lực hoặc bị thay thế. Trả lời theo quy định còn hiệu lực, "
                 "sau đó BẮT BUỘC thêm mục \"⚠️ Thay đổi pháp luật\" nêu quy định cũ (văn bản, nội dung chính), "
                 "quy định mới (văn bản, nội dung chính) và thời điểm thay đổi.")
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
