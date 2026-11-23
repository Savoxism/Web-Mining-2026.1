"""Grounded answer generation with a self-hosted Qwen3-4B (vLLM, OpenAI-compatible API)."""

from __future__ import annotations

import os
import re

import pandas as pd

from lexgraph.query import expand_query

MONEY = re.compile(r"(\d{1,3}(?:\.\d{3})+)\s*đồng")
TOK = re.compile(r"\w+", re.UNICODE)
FACTS_HEADER = "THAY ĐỔI ĐÃ XÁC MINH"
# a change fact is kept only if the new clause is about as relevant to the question as the old one:
# CE(q, new) >= CE(q, old) - NOTICE_DELTA (cross-encoder logits); tuned on validation questions
# (scripts/eval/tune_notice_gate.py). A wrong version alignment gives an unrelated new clause and a low CE(q, new).
NOTICE_DELTA = float(os.environ.get("LEXGRAPH_NOTICE_DELTA", "inf"))
# the clause counts as changed when the fine amounts differ or the token Jaccard of old vs new clause is below this
NOTICE_JACCARD = float(os.environ.get("LEXGRAPH_NOTICE_JACCARD", "0.6"))

# optional second check by the LLM itself: does the answer to *this question* differ between the two clauses?
CHECK_PROMPT = """Câu hỏi: {q}

Quy định cũ:
{old}

Quy định mới:
{new}

Theo quy định mới, câu trả lời cho câu hỏi trên có khác so với theo quy định cũ không (khác mức tiền, thời hạn, điều kiện,
đối tượng, cơ quan có thẩm quyền hoặc thủ tục)? Nếu quy định mới không nói về nội dung của câu hỏi thì trả lời KHÔNG.
Chỉ trả lời một từ: CÓ hoặc KHÔNG."""
_CHECKER = None


def set_fact_checker(client, model: str | None, thinking: bool = False) -> None:
    """Enable (client, model) or disable (client=None) the LLM check of change facts."""
    global _CHECKER
    _CHECKER = None if client is None else (client, model, thinking)


def llm_says_changed(client, model: str, question: str, old: str, new: str, thinking: bool = False) -> bool:
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=2048 if thinking else 8,
        messages=[{"role": "user", "content": CHECK_PROMPT.format(q=question, old=old[:2500], new=new[:2500])}],
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    out = (r.choices[0].message.content or "").split("</think>")[-1].strip().upper()
    return out.startswith("CÓ") or out.startswith("CO")
NOTICE_RE = re.compile(r"\n*\s*(?:\*\*)?⚠️?\s*(?:\*\*)?\s*Thay đổi pháp luật.*", re.S)

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp pháp luật Việt Nam.
Quy tắc:
1. Chỉ trả lời dựa trên các trích đoạn văn bản pháp luật được cung cấp. Không dùng kiến thức bên ngoài.
2. Mỗi ý quan trọng phải có trích dẫn dạng [số] tương ứng với trích đoạn, kèm số hiệu văn bản và điều, khoản.
3. Trả lời theo các trích đoạn CÒN HIỆU LỰC tại thời điểm hỏi. Chỉ thêm mục "⚠️ Thay đổi pháp luật" khi có mục
   "THAY ĐỔI ĐÃ XÁC MINH", và chỉ nêu đúng các thông tin trong mục đó (quy định cũ, quy định mới, thời điểm thay đổi).
4. Nếu các trích đoạn không đủ căn cứ để trả lời, nói rõ "Không tìm thấy đủ căn cứ pháp lý" và không suy đoán.
5. Trả lời ngắn gọn, rõ ràng, bằng tiếng Việt."""


def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and x != x) else str(x)


def _fmt_date(x) -> str:
    return pd.Timestamp(x).strftime("%d/%m/%Y") if pd.notna(x) else "không rõ"


def _body_tokens(text: str) -> set:
    lines = [l for l in str(text).split("\n") if l.strip()]
    return set(TOK.findall(" ".join(lines[1:]).lower())) if len(lines) > 1 else set(TOK.findall(str(text).lower()))


def change_fact(lg, old: dict, q_vec, q_exp: str, t: pd.Timestamp) -> dict | None:
    """Does the clause the question is about really differ between the old article and its successor?

    Returns a fact record when it does (different fine amounts, or clearly different wording), else None."""
    succ_rows = lg.rows_of_aid.get(_s(old.get("successor_aid")), [])
    if not succ_rows:
        return None
    old_clause = lg.relevant_text(q_vec, _s(old.get("aid")), str(old.get("text", "")), n=1, query_text=q_exp)
    old_ce = new_ce = float("nan")
    if lg.ce is not None:
        import numpy as np
        old_row = old.get("row")
        rows = ([int(old_row)] if old_row is not None else []) + list(succ_rows)
        raw = np.asarray(lg.ce_scores(q_exp, np.array(rows)), dtype=float)
        if old_row is not None:
            old_ce, raw = float(raw[0]), raw[1:]
        sc = raw.copy()
        if q_exp and "(" in q_exp:  # exact legal phrases from query normalisation win (same rule as clause selection)
            terms = [x.strip().lower() for x in q_exp.rsplit("(", 1)[1].rstrip(")").split(";") if x.strip()]
            bonus = np.array([sum(tm in str(lg.corpus.text.iat[r]).lower() for tm in terms) for r in succ_rows], float)
            sc = sc + (np.abs(sc).max() + 1.0) * bonus
        k = int(sc.argmax())
        new_row, new_ce = succ_rows[k], float(raw[k])
    else:
        new_row = succ_rows[0]
    new = lg.corpus.iloc[new_row]
    new_clause = str(new.text)
    mo, mn = MONEY.findall(old_clause), MONEY.findall(new_clause)
    a, b = _body_tokens(old_clause), _body_tokens(new_clause)
    jac = len(a & b) / max(1, len(a | b))
    money_changed = bool(mo and mn and set(mo[:4]) != set(mn[:4]))
    if not (money_changed or jac < NOTICE_JACCARD):
        return None
    art = lambda aid: aid.split("#")[-1].split("~")[0] if "#" in aid else "?"  # noqa: E731
    return {"old_doc": _s(old.get("so_ky_hieu")), "old_art": art(_s(old.get("aid"))),
            "old_amount": "–".join(mo[:2]) + " đồng" if mo else "",
            "new_doc": _s(new.so_ky_hieu), "new_art": art(_s(new.aid)),
            "new_amount": "–".join(mn[:2]) + " đồng" if mn else "",
            "since": _fmt_date(new.valid_from), "new_row": int(new_row), "old_ce": old_ce, "new_ce": new_ce,
            "money_changed": money_changed, "jaccard": round(jac, 3), "old_clause": old_clause[:2500],
            "new_clause": new_clause[:2500]}


def _fact_line(f: dict) -> str:
    old = f"{f['old_doc']}, Điều {f['old_art']}" + (f" (mức {f['old_amount']})" if f["old_amount"] else "")
    new = f"{f['new_doc']}, Điều {f['new_art']}" + (f" (mức {f['new_amount']})" if f["new_amount"] else "")
    return f"- Quy định cũ: {old}. Quy định mới: {new}, hiệu lực từ {f['since']}."


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
        n_old, facts, fact_keys = 0, [], set()
        for r in rows:
            if n_old >= 1 or not (r["stale"] and _s(r.get("successor_aid"))):
                continue
            # an old version is relevant if its successor is already a source, or it scores close to the best source
            if r["successor_aid"] in chosen_aids or r["ce_score"] >= top_ce - 1.5:
                f = change_fact(lg, r, q_vec, q_exp, t) if q_vec is not None else None
                if f is not None and f["new_ce"] == f["new_ce"] and f["old_ce"] == f["old_ce"] \
                        and f["new_ce"] < f["old_ce"] - NOTICE_DELTA:
                    f = None  # the new clause is not about the question: probably a wrong alignment
                if f is not None and _CHECKER is not None and query and \
                        not llm_says_changed(_CHECKER[0], _CHECKER[1], q_exp, f["old_clause"], f["new_clause"], _CHECKER[2]):
                    f = None  # the wording differs but the answer to this question does not
                if f is not None and (f["old_doc"], f["old_art"], f["new_doc"], f["new_art"]) not in fact_keys:
                    fact_keys.add((f["old_doc"], f["old_art"], f["new_doc"], f["new_art"]))
                    facts.append(f)
                    chosen.append(r)
                    n_old += 1
                succ_row = f["new_row"] if f is not None else (lg.rows_of_aid.get(r["successor_aid"], [None])[0])
                if succ_row is not None and r["successor_aid"] not in chosen_aids:
                    s = lg.corpus.iloc[succ_row].to_dict()
                    s.update({"stale": False, "row": succ_row})
                    chosen.append(s)
                    chosen_aids.add(r["successor_aid"])
    else:
        chosen = rows[:n_sources]
        facts = []

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
            text = lg.relevant_text(q_vec, aid, text, n=1, per_chunk_chars=2 * max_chars, query_text=q_exp)
        else:
            text = text[:max_chars]
        versions = getattr(lg, "versions", None)
        if time_aware and versions is not None and not r.get("stale") and aid and q_vec is not None:
            am = versions.amendments(aid, t)
            am = am[am.new_text.notna()]
            # only amendments of the clauses shown (or of the whole article)
            shown = set(re.findall(r"(?m)^(\d+[a-zđ]?)\.\s", text))
            if shown:
                am = am[am.target_clause.isna() | am.target_clause.astype(str).isin(shown)]
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
    ctx = "\n\n".join(blocks)
    if time_aware and facts:
        ctx += f"\n\n{FACTS_HEADER} (dựa trên khoản liên quan tới câu hỏi):\n" + "\n".join(_fact_line(f) for f in facts)
    return ctx, sources


def answer(client, model: str, question: str, context: str, t: pd.Timestamp, thinking: bool = False,
           max_tokens: int = 1024) -> str:
    user = (f"Thời điểm hỏi: {_fmt_date(t)}\n\nCác trích đoạn văn bản pháp luật:\n{context}\n\n"
            f"Câu hỏi: {expand_query(question)}")
    has_facts = FACTS_HEADER in context
    if has_facts:
        user += ("\n\nLưu ý: có mục THAY ĐỔI ĐÃ XÁC MINH. Trả lời theo quy định còn hiệu lực, sau đó thêm mục "
                 "\"⚠️ Thay đổi pháp luật\" chỉ dựa đúng vào mục đó.")
    else:
        user += "\n\nLưu ý: không có thay đổi pháp luật đã xác minh; KHÔNG viết mục \"⚠️ Thay đổi pháp luật\"."
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        temperature=0.1 if not thinking else 0.6,
        top_p=0.8 if not thinking else 0.95,
        max_tokens=max_tokens if not thinking else 4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}, "top_k": 20},
    )
    text = resp.choices[0].message.content or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return verify_notice(text.strip(), context)


def verify_notice(text: str, context: str) -> str:
    """Keep a change notice only when it is backed by verified change facts; add one if the model forgot it."""
    if FACTS_HEADER not in context:
        return NOTICE_RE.sub("", text).strip()
    if "Thay đổi pháp luật" not in text:
        facts = context.split(FACTS_HEADER, 1)[1].split("\n", 1)[1].strip()
        text += "\n\n⚠️ Thay đổi pháp luật:\n" + facts
    return text
