"""Answer-quality evaluation of Qwen3-4B in three settings, judged by a larger local LLM.

Settings: no retrieval, plain RAG (top sources, no validity labels), time-aware RAG (ours).
Questions: stable test questions, temporal test questions (gold superseded at t_eval) and
out-of-scope questions that should be refused.

Metrics
  * citation accuracy: answer cites [k] and at least one cited source is a current gold article;
  * correctness / faithfulness: judged by the judge LLM against the reference article (current law)
    and against the sources given to the model;
  * change-notice rate: share of temporal answers with a "Thay đổi pháp luật" section
    (and false-alarm rate on stable questions);
  * correct refusals: share of out-of-scope questions answered with "Không tìm thấy đủ căn cứ".

Also audits successor alignments: the judge decides whether the proposed new article is the updated
version of the old one.

Usage (needs the Qwen3-4B server and the judge server):
    .venv/bin/python scripts/eval/eval_answers.py --bi <bi> --ce <ce> --weights outputs/.../results.json \
        --llm-url http://127.0.0.1:8100/v1 --judge-url http://127.0.0.1:8200/v1 --out outputs/.../answers
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lexgraph.llm import SYSTEM_PROMPT, answer, build_context  # noqa: E402
from lexgraph.pipeline import LexGraph, Weights  # noqa: E402

OUT_OF_SCOPE = [
    "Luật thuế thu nhập cá nhân của Singapore quy định mức thuế suất cao nhất là bao nhiêu?",
    "Ai là cầu thủ ghi nhiều bàn thắng nhất lịch sử World Cup?",
    "Công thức nấu phở bò Hà Nội gồm những nguyên liệu gì?",
    "Hiến pháp Hoa Kỳ có bao nhiêu tu chính án?",
    "Giá vàng SJC hôm nay là bao nhiêu?",
    "Làm thế nào để cài đặt Python trên máy tính Windows?",
    "Quy định về giới hạn tốc độ trên đường cao tốc ở Đức là gì?",
    "Thủ tục xin visa du lịch Nhật Bản cần những giấy tờ gì theo quy định của Nhật?",
    "Luật lao động của Pháp quy định bao nhiêu giờ làm việc mỗi tuần?",
    "Kết quả trận chung kết Champions League năm 2024 là gì?",
    "Tác giả của tiểu thuyết Chiến tranh và hòa bình là ai?",
    "Làm sao để giảm cân nhanh trong một tháng?",
    "Bang California quy định mức lương tối thiểu là bao nhiêu?",
    "Thuế giá trị gia tăng ở Thái Lan hiện nay là bao nhiêu phần trăm?",
    "Cách tính diện tích hình tròn như thế nào?",
    "Quy định về sở hữu súng cá nhân ở Mỹ ra sao?",
    "Thời tiết Hà Nội ngày mai thế nào?",
    "Liên minh châu Âu quy định gì về bảo vệ dữ liệu cá nhân trong GDPR?",
    "Nên đầu tư vào cổ phiếu hay trái phiếu trong năm nay?",
    "Luật hôn nhân của Hàn Quốc quy định độ tuổi kết hôn tối thiểu là bao nhiêu?",
]

NO_RETRIEVAL_PROMPT = """Bạn là trợ lý hỏi đáp pháp luật Việt Nam. Trả lời ngắn gọn bằng tiếng Việt và nêu căn cứ pháp lý nếu biết.
Nếu câu hỏi không thuộc pháp luật Việt Nam hoặc bạn không chắc chắn, nói rõ "Không tìm thấy đủ căn cứ pháp lý"."""

JUDGE_QA = """You are grading a Vietnamese legal assistant. Reply with a JSON object only.
Question: {q}

Reference article (the law in force at {t}):
{ref}

Sources shown to the assistant:
{ctx}

Assistant answer:
{ans}

Return {{"correct": 0 or 1, "faithful": 0 or 1}} where
- correct = 1 if the answer's main legal content agrees with the reference article (the law in force),
- faithful = 1 if every legal claim in the answer is supported by the sources shown (if no sources were shown, faithful = 1 only when the answer makes no unsupported specific claims)."""

JUDGE_ALIGN = """You are checking a legal version alignment. Reply with a JSON object only.
Old article:
{old}

Proposed new article (from a later amending/replacing document):
{new}

Return {{"same_provision": 0 or 1}}: 1 if the new article regulates the same legal matter as the old one, so that it
updates, replaces or amends it; 0 otherwise."""


JUDGE_CHANGED = """You are comparing two versions of a Vietnamese legal provision for one question. Reply with a JSON object only.
Question: {q}

Old version (in force earlier):
{old}

Current version (in force at {t}):
{new}

Return {{"changed": 0 or 1}}: 1 if the answer to the question differs between the two versions (e.g. a different amount,
time limit, condition, competent authority or procedure); 0 if the answer is the same in substance."""


def chat(client, model, prompt, max_tokens=512):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               temperature=0.0, max_tokens=max_tokens,
                                               extra_body={"reasoning_effort": "low"})
            return r.choices[0].message.content or ""
        except Exception as e:  # retry on transient server errors
            err = e
            time.sleep(2 * (attempt + 1))
    return f"ERROR {err}"


def parse_json(text, keys):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    return {k: int(d.get(k, 0)) if str(d.get(k, "0")).isdigit() else 0 for k in keys}


def cited(ans: str) -> set[int]:
    return {int(x) for x in re.findall(r"\[(\d+)\]", ans)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", required=True)
    ap.add_argument("--ce", required=True)
    ap.add_argument("--weights", required=True, help="results.json of eval_pipeline (best_val weights)")
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--t-eval", default="2026-11-23")
    ap.add_argument("--n-stable", type=int, default=100)
    ap.add_argument("--n-temporal", type=int, default=50)
    ap.add_argument("--n-align", type=int, default=100)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--llm-model", default="qwen3-4b")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.Timestamp(args.t_eval)
    rng = np.random.default_rng(42)

    best = json.loads(Path(args.weights).read_text())["best_val"]
    w = Weights(beta=best["beta"], gamma=best["gamma"], mu=best["mu"], inject=True)
    w_plain = Weights(inject=False, static=True)
    lg = LexGraph(args.bi, args.ce)
    llm = OpenAI(base_url=args.llm_url, api_key="EMPTY")
    judge = OpenAI(base_url=args.judge_url, api_key="EMPTY")

    # ---------------- question sample
    test = pd.read_parquet(Path(args.data) / "queries_test.parquet").sample(frac=1.0, random_state=42)
    stable, temporal = [], []
    for q in test.itertuples(index=False):
        gold = sorted(lg.row_of[c] for c in q.cids if c in lg.row_of)
        if not gold or lg.doc_ids[gold[0]] == "":
            continue
        st, _, su = lg.status(np.array(gold), t, w.tau)
        if st.any() and any(s is not None and lg.rows_of_aid.get(s.succ_aid) for s in su):
            if len(temporal) < args.n_temporal:
                cur = set()
                for g, a, s in zip(gold, st, su):
                    cur |= set(lg.rows_of_aid[s.succ_aid]) if (a and s is not None and lg.rows_of_aid.get(s.succ_aid)) else {g}
                temporal.append({"query": q.query, "gold": gold, "cur": sorted(cur), "kind": "temporal"})
        elif not st.any() and len(stable) < args.n_stable:
            stable.append({"query": q.query, "gold": gold, "cur": gold, "kind": "stable"})
        if len(stable) >= args.n_stable and len(temporal) >= args.n_temporal:
            break
    items = stable + temporal + [{"query": q, "gold": [], "cur": [], "kind": "out_of_scope"} for q in OUT_OF_SCOPE]
    print(f"{len(stable)} stable, {len(temporal)} temporal, {len(OUT_OF_SCOPE)} out-of-scope", flush=True)

    # ---------------- retrieval + contexts (sequential, GPU)
    for it in items:
        r_ours = lg.search(it["query"], t, w, k_rerank=30, top=20)
        r_plain = lg.search(it["query"], t, w_plain, k_rerank=30, top=15)
        it["ctx_ours"], it["src_ours"] = build_context(r_ours, lg, t, True, query=it["query"])
        it["ctx_plain"], it["src_plain"] = build_context(r_plain, lg, t, False, query=it["query"])

    # ---------------- generation (parallel requests)
    def gen(job):
        it, setting = job
        if setting == "none":
            r = llm.chat.completions.create(
                model=args.llm_model, temperature=0.3, top_p=0.8, max_tokens=1024,
                messages=[{"role": "system", "content": NO_RETRIEVAL_PROMPT}, {"role": "user", "content": it["query"]}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20})
            txt = (r.choices[0].message.content or "").split("</think>")[-1].strip()
        else:
            txt = answer(llm, args.llm_model, it["query"], it[f"ctx_{setting}"], t)
        it[f"ans_{setting}"] = txt

    jobs = [(it, s) for it in items for s in ["none", "plain", "ours"]]
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(gen, jobs))
    print("generated", flush=True)

    # ---------------- judging
    def ref_text(it):
        return "\n\n".join(str(lg.corpus.text.iat[r])[:2500] for r in it["cur"][:2]) or "(none)"

    def jqa(job):
        it, s = job
        ctx = it.get(f"ctx_{s}", "(no sources)") if s != "none" else "(no sources)"
        txt = chat(judge, args.judge_model, JUDGE_QA.format(q=it["query"], t=args.t_eval, ref=ref_text(it),
                                                            ctx=ctx[:9000], ans=it[f"ans_{s}"][:4000]))
        it[f"judge_{s}"] = parse_json(txt, ["correct", "faithful"])

    in_scope = [it for it in items if it["kind"] != "out_of_scope"]
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(jqa, [(it, s) for it in in_scope for s in ["none", "plain", "ours"]]))

    # did the answer to a temporal question really change? (a notice is only expected then)
    def jchanged(it):
        old = "\n\n".join(str(lg.corpus.text.iat[r])[:2500] for r in it["gold"][:2])
        txt = chat(judge, args.judge_model, JUDGE_CHANGED.format(q=it["query"], old=old, new=ref_text(it), t=args.t_eval),
                   max_tokens=256)
        it["changed"] = parse_json(txt, ["changed"])["changed"]

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(jchanged, [it for it in in_scope if it["kind"] == "temporal"]))

    # ---------------- metrics
    def has_notice(a):
        return ("thay đổi pháp luật" in a.lower()) or ("⚠" in a)

    def refused(a):
        return "không tìm thấy đủ căn cứ" in a.lower()

    rows = []
    for s, label in [("none", "Qwen3-4B, no retrieval"), ("plain", "Qwen3-4B + plain RAG"), ("ours", "Qwen3-4B + time-aware RAG (ours)")]:
        cit = []
        if s != "none":
            for it in in_scope:
                src = {x["k"]: x["row"] for x in it[f"src_{s}"]}
                c = cited(it[f"ans_{s}"])
                cit.append(float(any(src.get(k) in set(it["cur"]) for k in c)))
        tmp = [it for it in in_scope if it["kind"] == "temporal"]
        stb = [it for it in in_scope if it["kind"] == "stable"]
        oos = [it for it in items if it["kind"] == "out_of_scope"]
        rows.append({
            "setting": label,
            "citation_acc": round(float(np.mean(cit)), 3) if cit else None,
            "correct": round(float(np.mean([it[f"judge_{s}"]["correct"] for it in in_scope])), 3),
            "correct_stable": round(float(np.mean([it[f"judge_{s}"]["correct"] for it in stb])), 3),
            "correct_temporal": round(float(np.mean([it[f"judge_{s}"]["correct"] for it in tmp])), 3),
            "faithful": round(float(np.mean([it[f"judge_{s}"]["faithful"] for it in in_scope])), 3),
            "change_notice_temporal": round(float(np.mean([has_notice(it[f"ans_{s}"]) for it in tmp])), 3),
            "change_notice_false_alarm_stable": round(float(np.mean([has_notice(it[f"ans_{s}"]) for it in stb])), 3),
            "change_notice_recall_changed": round(float(np.mean([has_notice(it[f"ans_{s}"]) for it in tmp if it["changed"]] or [0])), 3),
            "change_notice_on_unchanged_temporal": round(float(np.mean([has_notice(it[f"ans_{s}"]) for it in tmp if not it["changed"]] or [0])), 3),
            "correct_refusals_oos": round(float(np.mean([refused(it[f"ans_{s}"]) for it in oos])), 3),
            "over_refusals_in_scope": round(float(np.mean([refused(it[f"ans_{s}"]) for it in in_scope])), 3),
        })
    metrics = {"answers": rows, "n_stable": len(stable), "n_temporal": len(temporal), "n_out_of_scope": len(OUT_OF_SCOPE),
               "n_temporal_changed": int(sum(it["changed"] for it in temporal)), "successors": lg.succ_version}

    # ---------------- alignment audit
    succ = pd.read_parquet(Path("data/kg/temporal") / os.environ.get("LEXGRAPH_SUCCESSORS", "successors.parquet"))
    cand = succ[succ.accept] if lg.succ_version == "v2" else succ[(succ["rank"] == 0) & ((succ.sim >= w.tau) | succ.explicit)]
    cand = cand[cand.succ_aid.map(lambda a: bool(lg.rows_of_aid.get(a)))]
    sample = cand.sample(min(args.n_align, len(cand)), random_state=42)

    def jalign(r):
        old = str(lg.corpus.text.iat[lg.row_of[int(r.cid)]])[:2500]
        new = "\n".join(str(lg.corpus.text.iat[x]) for x in lg.rows_of_aid[r.succ_aid][:3])[:3000]
        txt = chat(judge, args.judge_model, JUDGE_ALIGN.format(old=old, new=new), max_tokens=256)
        return parse_json(txt, ["same_provision"])["same_provision"]

    with ThreadPoolExecutor(args.workers) as ex:
        ok = list(ex.map(jalign, sample.itertuples(index=False)))
    sample = sample.assign(judge_same=ok)
    sample.to_parquet(out / "alignment_audit.parquet", index=False)
    metrics["alignment_audit"] = {
        "n": len(sample), "accuracy": round(float(np.mean(ok)), 3),
        "accuracy_explicit": round(float(sample[sample.explicit].judge_same.mean()), 3) if sample.explicit.any() else None,
        "accuracy_similarity_only": round(float(sample[~sample.explicit].judge_same.mean()), 3) if (~sample.explicit).any() else None,
        "n_explicit": int(sample.explicit.sum()),
    }
    (out / "answers.json").write_text(json.dumps([{k: v for k, v in it.items() if not k.startswith("src_")} | {
        "src_ours": it.get("src_ours"), "src_plain": it.get("src_plain")} for it in items], ensure_ascii=False, indent=1, default=str))
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
