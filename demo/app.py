"""LexGraph-VN Streamlit demo (English interface).

One question box and one date: the answer follows the law in force on that date. When the rule the question is about
changed, a short verified note shows what changed and since when.

Run on the GPU server (models stay in GPU memory), then tunnel the port:
    LEXGRAPH_BI=outputs/models/bge-m3-legal/final LEXGRAPH_CE=outputs/models/reranker-legal/final \
    .venv/bin/streamlit run demo/app.py --server.port 8501 --server.address 127.0.0.1
    ssh -L 8501:localhost:8501 H200_ANNP36     # then open http://localhost:8501
"""

import html
import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from lexgraph.llm import answer, build_context, set_fact_checker, to_vietnamese  # noqa: E402
from lexgraph.pipeline import LexGraph, Weights  # noqa: E402
from lexgraph.versions import VersionIndex, word_diff_html  # noqa: E402

BI = os.environ.get("LEXGRAPH_BI", "BAAI/bge-m3")
BI_REV = os.environ.get("LEXGRAPH_BI_REV") or None
CE = os.environ.get("LEXGRAPH_CE", "BAAI/bge-reranker-v2-m3")
CE_REV = os.environ.get("LEXGRAPH_CE_REV") or None
LLM_URL = os.environ.get("LEXGRAPH_LLM_URL", "http://127.0.0.1:8100/v1")
LLM_MODEL = os.environ.get("LEXGRAPH_LLM_MODEL", "qwen3-4b")
# data/kg is the snapshot; data/kg_live adds the documents crawled from vbpl.vn (scripts/kg/update_kg.py)
KG_DIR = Path(os.environ.get("LEXGRAPH_KG_DIR", "data/kg"))
GAMMA, K_RERANK, N_SOURCES = 2.0, 30, 5  # tuned on validation

DOC_TYPES = {"Bộ luật": "Code", "Luật": "Law", "Pháp lệnh": "Ordinance", "Nghị định": "Decree", "Nghị quyết": "Resolution",
             "Nghị quyết liên tịch": "Joint Resolution", "Thông tư": "Circular", "Thông tư liên tịch": "Joint Circular",
             "Quyết định": "Decision", "Lệnh": "Order", "Sắc lệnh": "Edict", "Hiến pháp": "Constitution"}
MONEY = re.compile(r"(\d{1,3}(?:\.\d{3})+)")

st.set_page_config(page_title="LexGraph-VN", page_icon="⚖️", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.change { border-left: 4px solid #b7791f; padding: .6rem .9rem; margin: .8rem 0 .4rem; background: rgba(183,121,31,.07);
  border-radius: 4px; }
.src { font-size: .92rem; margin: .15rem 0; }
.ok { color: #2f7d4f; } .old { color: #8a8f98; }
.muted { color: #8a8f98; font-size: .85rem; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models and the legal knowledge graph…")
def load():
    lg = LexGraph(BI, CE, bi_revision=BI_REV, ce_revision=CE_REV, kg_dir=str(KG_DIR),
                  temporal_dir=str(KG_DIR / "temporal"))
    lg.versions = VersionIndex(str(KG_DIR / "amendments.parquet"))
    return lg


def llm():
    """OpenAI-compatible client for Qwen3-4B on vLLM, or None when the server is down."""
    from openai import OpenAI
    client = OpenAI(base_url=LLM_URL, api_key="EMPTY", timeout=120, max_retries=0)
    try:
        client.models.list()
    except Exception:
        set_fact_checker(None, None)
        return None
    set_fact_checker(client, LLM_MODEL, thinking=False)  # Qwen3-4B confirms each change fact
    return client


def fmt_date(x) -> str:
    return pd.Timestamp(x).strftime("%d/%m/%Y") if pd.notna(x) else "?"


def doc_name(so_ky_hieu: str, loai: str | None = None) -> str:
    if not loai:
        m = lg.docs[lg.docs.so_ky_hieu == so_ky_hieu]
        loai = m.loai_van_ban.iloc[0] if len(m) else ""
    return f"{DOC_TYPES.get(loai, '')} {so_ky_hieu}".strip()


def art_of(aid) -> str:
    return str(aid).split("#")[-1].split("~")[0] if isinstance(aid, str) and "#" in aid else "?"


def vnd(amount: str) -> str:
    """'800.000–1.000.000 đồng' -> '800,000 to 1,000,000 VND'."""
    nums = [n.replace(".", ",") for n in MONEY.findall(amount or "")]
    return " to ".join(nums) + " VND" if nums else ""


# ------------------------------------------------------------------ page
st.title("⚖️ LexGraph-VN")
st.caption("Questions about Vietnamese law, answered under the law in force on the date you choose.")

lg = load()

with st.form("ask"):
    question = st.text_area("Question", height=90, label_visibility="collapsed",
                            placeholder="e.g. How much is the fine for running a red light on a motorbike?")
    c1, c2 = st.columns([1, 2])
    t_q = pd.Timestamp(c1.date_input("Law as of", value=date.today(), format="DD/MM/YYYY"))
    go = c2.form_submit_button("Ask", type="primary")

if go and question.strip():
    client = llm()
    q_vi = question
    if client is not None:
        with st.spinner("Reading the question…"):
            try:
                q_vi = to_vietnamese(client, LLM_MODEL, question)
            except Exception:
                client = None
    w = Weights(beta=0.0, gamma=GAMMA, mu=0.0, tau=0.80, use_ce=True, inject=True)
    with st.spinner("Searching the law in force…"):
        res = lg.search(q_vi, t_q, w, k_rerank=K_RERANK, top=20)
    facts, ans, sources = [], None, []
    if client is not None:
        with st.spinner("Writing the answer…"):
            try:
                ctx, sources, facts = build_context(res, lg, t_q, True, n_sources=N_SOURCES, query=q_vi, return_facts=True)
                ans = answer(client, LLM_MODEL, question, ctx, t_q, lang="en", question_vi=q_vi)
            except Exception:
                ans, facts, sources = None, [], []
    st.session_state["last"] = dict(question=question, t=t_q, res=res, facts=facts, ans=ans, sources=sources)

if "last" in st.session_state:
    L = st.session_state["last"]
    res, t_used = L["res"], L["t"]

    st.subheader("Answer")
    if L["ans"]:
        st.markdown(L["ans"])
    else:
        st.info("The answer model is offline, so only the retrieved articles are shown.")
    st.markdown(f"<div class='muted'>Under the law in force on {fmt_date(t_used)}.</div>", unsafe_allow_html=True)

    for f in L["facts"][:1]:  # the change that matters for this question, verified clause by clause
        old_amt, new_amt = vnd(f["old_amount"]), vnd(f["new_amount"])
        amounts = (f"<br>Before: <s>{old_amt}</s> &nbsp;→&nbsp; now: <b>{new_amt}</b>" if old_amt and new_amt and old_amt != new_amt else "")
        st.markdown(
            f"<div class='change'><b>This rule changed on {f['since']}.</b> "
            f"{html.escape(doc_name(f['new_doc']))}, Art. {f['new_art']} replaced "
            f"{html.escape(doc_name(f['old_doc']))}, Art. {f['old_art']}.{amounts}</div>", unsafe_allow_html=True)
        with st.expander("Compare the clause before and after (Vietnamese)"):
            st.markdown(word_diff_html(f["old_clause"], f["new_clause"]), unsafe_allow_html=True)

    st.subheader("Sources")
    if L["sources"]:
        rows = [lg.corpus.iloc[s["row"]] if s["row"] is not None else None for s in L["sources"]]
        items = [(s["k"], r, s["stale"]) for s, r in zip(L["sources"], rows) if r is not None]
    else:
        top = res[~res.stale].assign(_a=res.aid.astype(str).str.split("~").str[0]).drop_duplicates("_a").head(N_SOURCES)
        items = [(k, lg.corpus.iloc[int(r.row)], False) for k, r in enumerate(top.itertuples(), start=1)]
    for k, r, stale in items:
        name = doc_name(str(r.so_ky_hieu), r.loai_van_ban) if isinstance(r.so_ky_hieu, str) else "Unlinked passage"
        if stale:
            status = f"<span class='old'>older version, no longer in force since {fmt_date(r.valid_to)}</span>" \
                if pd.notna(r.valid_to) else "<span class='old'>older version</span>"
        else:
            status = f"<span class='ok'>in force since {fmt_date(r.valid_from)}</span>"
        st.markdown(f"<div class='src'>[{k}] <b>{html.escape(name)}</b>, Art. {art_of(r.aid)} · {status}</div>",
                    unsafe_allow_html=True)
        with st.expander("Original text (Vietnamese)", expanded=False):
            st.text(str(r.text)[:3000])
