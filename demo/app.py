"""LexGraph Streamlit demo.

Run on the GPU server (models stay in GPU memory), then tunnel the port:
    LEXGRAPH_BI=BAAI/bge-m3 LEXGRAPH_CE=BAAI/bge-reranker-v2-m3 \
    .venv/bin/streamlit run demo/app.py --server.port 8501 --server.address 127.0.0.1
    ssh -L 8501:localhost:8501 H200_ANNP36     # then open http://localhost:8501
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from lexgraph.llm import answer, build_context  # noqa: E402
from lexgraph.pipeline import LexGraph, Weights  # noqa: E402

BI = os.environ.get("LEXGRAPH_BI", "BAAI/bge-m3")
BI_REV = os.environ.get("LEXGRAPH_BI_REV") or None
CE = os.environ.get("LEXGRAPH_CE", "BAAI/bge-reranker-v2-m3")
CE_REV = os.environ.get("LEXGRAPH_CE_REV") or None
LLM_URL = os.environ.get("LEXGRAPH_LLM_URL", "http://127.0.0.1:8100/v1")
LLM_MODEL = os.environ.get("LEXGRAPH_LLM_MODEL", "qwen3-4b")

EXAMPLES = {
    "Câu hỏi thông thường": "Người lao động làm việc trong điều kiện bình thường được nghỉ hằng năm bao nhiêu ngày?",
    "Câu hỏi về luật đã thay đổi": "Người điều khiển xe máy vượt đèn đỏ bị phạt bao nhiêu tiền?",
}

st.set_page_config(page_title="LexGraph", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình và knowledge graph…")
def load():
    return LexGraph(BI, CE, bi_revision=BI_REV, ce_revision=CE_REV)


@st.cache_resource
def llm_client():
    from openai import OpenAI
    return OpenAI(base_url=LLM_URL, api_key="EMPTY")


def badge(stale: bool) -> str:
    return "🔴 hết hiệu lực / bị thay thế" if stale else "🟢 còn hiệu lực"


def fmt_date(x) -> str:
    return pd.Timestamp(x).strftime("%d/%m/%Y") if pd.notna(x) else "—"


def graph_html(lg: LexGraph, doc_ids: list[str], t: pd.Timestamp) -> str:
    from pyvis.network import Network
    from lexgraph.temporal import in_force

    edges = st.session_state.get("_edges")
    if edges is None:
        edges = pd.read_parquet("data/kg/edges.parquet")
        edges = edges[edges.type.isin({"amends", "replaces", "repeals", "partially_repeals", "guides"})]
        st.session_state["_edges"] = edges
    focus = set(d for d in doc_ids if d)
    e = edges[edges.src.isin(focus) | edges.dst.isin(focus)]
    e = e[e.type.isin({"amends", "replaces", "repeals", "partially_repeals"}) | (e.src.isin(focus) & e.dst.isin(focus))]
    e = e.head(120)
    nodes = focus | set(e.src) | set(e.dst)
    net = Network(height="520px", width="100%", directed=True, bgcolor="#ffffff", font_color="#1d2330")
    docs = lg.docs
    for n in nodes:
        if n not in docs.index:
            continue
        d = docs.loc[n]
        ok = in_force([d.valid_from], [d.valid_to], t)[0]
        color = "#2f7d4f" if ok else "#b7791f"
        label = str(d.so_ky_hieu)
        title = f"{d.so_ky_hieu}\n{str(d.title)[:120]}\nHiệu lực: {fmt_date(d.valid_from)} → {fmt_date(d.valid_to)}"
        net.add_node(n, label=label, title=title, color=color, size=26 if n in focus else 14,
                     borderWidth=3 if n in focus else 1)
    style = {"amends": ("#1f5f8b", "sửa đổi"), "replaces": ("#9b2c2c", "thay thế"), "repeals": ("#9b2c2c", "bãi bỏ"),
             "partially_repeals": ("#c05621", "bãi bỏ 1 phần"), "guides": ("#718096", "hướng dẫn")}
    for r in e.itertuples(index=False):
        if r.src in docs.index and r.dst in docs.index:
            c, lab = style.get(r.type, ("#999", r.type))
            net.add_edge(r.src, r.dst, color=c, title=lab, label=lab, font={"size": 9})
    net.set_options('{"physics": {"barnesHut": {"springLength": 140}}, "edges": {"arrows": {"to": {"enabled": true}}}}')
    return net.generate_html(notebook=False)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚖️ LexGraph")
st.sidebar.caption("Trợ lý hỏi đáp pháp luật Việt Nam có theo dõi hiệu lực theo thời gian")
t_q = pd.Timestamp(st.sidebar.date_input("Thời điểm hỏi", value=date.today(), format="DD/MM/YYYY"))
time_aware = st.sidebar.toggle("Xét hiệu lực theo thời gian", value=True)
graph_on = st.sidebar.toggle("Tín hiệu đồ thị (PageRank, lan truyền)", value=True)
use_ce = st.sidebar.toggle("Cross-encoder re-ranking", value=True)
use_llm = st.sidebar.toggle("Sinh câu trả lời bằng Qwen3-4B", value=True)
thinking = st.sidebar.toggle("Qwen3 thinking mode", value=False)
with st.sidebar.expander("Trọng số re-ranking"):
    beta = st.slider("β (PageRank)", 0.0, 2.0, 0.25, 0.05)
    gamma = st.slider("γ (phạt văn bản hết hiệu lực)", 0.0, 10.0, 4.0, 0.5)
    mu = st.slider("μ (lan truyền đồ thị)", 0.0, 3.0, 0.5, 0.1)
    tau = st.slider("τ (ngưỡng khớp phiên bản)", 0.5, 0.95, 0.80, 0.01)
    k_rerank = st.slider("Số ứng viên re-rank", 10, 100, 30, 5)
    n_sources = st.slider("Số trích đoạn đưa vào LLM", 2, 10, 5, 1)
st.sidebar.caption(f"Bi-encoder: `{BI}`  \nCross-encoder: `{CE}`  \nLLM: `{LLM_MODEL}` @ {LLM_URL}")

lg = load()

tab_qa, tab_kg = st.tabs(["Hỏi đáp", "Knowledge graph"])

# ------------------------------------------------------------------ QA tab
with tab_qa:
    st.title("Hỏi đáp pháp luật theo thời gian")
    c1, c2, _ = st.columns([1, 1, 2])
    for col, (label, q) in zip([c1, c2], EXAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state["question"] = q
    question = st.text_area("Câu hỏi", key="question", height=90,
                            placeholder="Ví dụ: Người lao động được nghỉ hằng năm bao nhiêu ngày?")
    go = st.button("Hỏi", type="primary")

    if go and question.strip():
        w = Weights(beta=beta if graph_on else 0.0, gamma=gamma if time_aware else 0.0, mu=mu if graph_on else 0.0,
                    tau=tau, use_ce=use_ce, inject=time_aware)
        with st.spinner("Đang truy hồi và xếp hạng…"):
            res = lg.search(question, t_q, w, k_rerank=k_rerank, top=15)
        st.session_state["last"] = (question, res, t_q, time_aware)

    if "last" in st.session_state:
        question, res, t_used, ta = st.session_state["last"]

        if use_llm:
            ctx, sources = build_context(res, lg, t_used, ta, n_sources=n_sources)
            with st.spinner("Qwen3-4B đang soạn câu trả lời…"):
                try:
                    ans = answer(llm_client(), LLM_MODEL, question, ctx, t_used, thinking=thinking)
                except Exception as ex:  # LLM server down: show retrieval only
                    ans = f"_Không gọi được LLM ({type(ex).__name__}). Kết quả truy hồi vẫn hiển thị bên dưới._"
            st.subheader("Câu trả lời")
            st.markdown(ans)
            with st.expander("Ngữ cảnh đưa vào LLM"):
                st.text(ctx)

        # change notices: stale results with a successor article
        if ta:
            changed = res[res.stale & res.successor_aid.notna()].head(3)
            for r in changed.itertuples():
                new_rows = lg.rows_of_aid.get(r.successor_aid, [])
                if not new_rows:
                    continue
                new = lg.corpus.iloc[new_rows[0]]
                st.warning(f"⚠️ Luật đã thay đổi: {r.so_ky_hieu} (Điều {str(r.aid).split('#')[-1]}) đã bị thay thế bởi "
                           f"{new.so_ky_hieu} (Điều {str(new.aid).split('#')[-1].split('~')[0]}), hiệu lực từ "
                           f"{fmt_date(new.valid_from)}. Độ khớp phiên bản {r.successor_sim:.2f}"
                           f"{', có dẫn chiếu tường minh' if r.successor_explicit else ''}.")
                a, b = st.columns(2)
                a.markdown(f"**Trước — {r.so_ky_hieu}**")
                a.text(str(r.text)[:2500])
                b.markdown(f"**Sau — {new.so_ky_hieu}**")
                b.text(str(new.text)[:2500])

        st.subheader("Các điều luật được truy hồi")
        for i, r in enumerate(res.itertuples(), start=1):
            art = str(r.aid).split("#")[-1].split("~")[0] if isinstance(r.aid, str) else "?"
            head = (f"{i}. {r.so_ky_hieu if isinstance(r.so_ky_hieu, str) else '(chưa liên kết văn bản)'} — Điều {art} · "
                    f"{badge(bool(r.stale)) if ta else ''} · điểm {r.final:.2f}")
            with st.expander(head, expanded=i <= 2):
                st.caption(f"{str(r.title)[:200] if isinstance(r.title, str) else ''}  \n"
                           f"Hiệu lực: {fmt_date(r.valid_from)} → {fmt_date(r.valid_to)} · nguồn: {r.source}"
                           + (f"  \nLý do: {r.stale_reason}" if r.stale and ta else ""))
                st.text(str(r.text)[:3000])
                st.caption(f"bi-encoder {r.bi_score:.3f} · cross-encoder {r.ce_score:.2f} · PageRank z {r.pagerank_z:.2f} · "
                           f"lan truyền {r.propagation:.2f}")

        st.subheader("Đồ thị quan hệ giữa các văn bản")
        st.caption("Xanh: còn hiệu lực tại thời điểm hỏi · Vàng: hết hiệu lực. Mũi tên đi từ văn bản mới tới văn bản bị tác động.")
        components.html(graph_html(lg, res.doc_id.dropna().unique().tolist()[:8], t_used), height=540)

# ------------------------------------------------------------------ KG tab
with tab_kg:
    st.title("Knowledge graph pháp luật")
    stats = json.loads(Path("data/kg/kg_stats.json").read_text())
    a, b, c, d = st.columns(4)
    a.metric("Văn bản", f"{stats['docs_selected']:,}")
    b.metric("Quan hệ", f"{stats['edges']:,}")
    c.metric("Điều luật", f"{stats['articles']:,}")
    d.metric("Đoạn QA liên kết", f"{stats['passage_link_rate']:.1%}")
    st.markdown("**Loại quan hệ**")
    st.bar_chart(pd.Series(stats["edge_types"]).sort_values(ascending=False))
    st.markdown("**Văn bản có PageRank cao nhất**")
    st.dataframe(pd.DataFrame(stats["top_pagerank"]), use_container_width=True, hide_index=True)
    st.markdown("**Trích xuất dẫn chiếu, so với quan hệ của cổng vbpl.vn**")
    st.json(stats["xref_eval"])
