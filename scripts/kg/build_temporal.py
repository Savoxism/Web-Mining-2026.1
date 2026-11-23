"""Temporal layer of the KG: document validity, article version alignment and the extended corpus.

For every QA passage that belongs to a document D with newer amending/replacing/repealing documents
S(D), find the article of S(D) that supersedes it:
  * explicit: a document in S(D) cites the passage's article ("Điều X ... <số hiệu of D>"), and
  * semantic: cosine similarity between the passage and the article (best chunk) under a bi-encoder.
The top successor candidates are stored; the pipeline decides at query time which are in force.
Articles of successor documents are chunked and added to the retrieval corpus (extended corpus).

Usage (GPU):
    .venv/bin/python scripts/kg/build_temporal.py --model BAAI/bge-m3 --revision <sha> --out data/kg/temporal
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lexgraph.temporal import SUCCESSION_TYPES, chunk_article, doc_intervals  # noqa: E402

DOC_COLS = ["id", "so_ky_hieu", "title", "loai_van_ban", "tinh_trang_hieu_luc", "valid_from", "valid_to",
            "pagerank", "authority", "community"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="data/kg")
    ap.add_argument("--corpus", default="data/processed_v2/corpus.parquet")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", default="data/kg/temporal")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=300)
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kg = Path(args.kg)

    docs = doc_intervals(pd.read_parquet(kg / "docs.parquet"), pd.read_parquet(kg / "edges.parquet"))
    edges = pd.read_parquet(kg / "edges.parquet")
    articles = pd.read_parquet(kg / "articles.parquet", columns=["aid", "doc_id", "art_no", "heading", "text"])
    links = pd.read_parquet(kg / "passage_links.parquet")
    xref = pd.read_parquet(kg / "xref_article_edges.parquet")
    corpus = pd.read_parquet(args.corpus)
    issue = docs.set_index("id").ngay_ban_hanh
    has_arts = set(articles.doc_id)

    # ---------------- successors of each document (reverse succession edges, newer actors only)
    succ_e = edges[edges.type.isin(SUCCESSION_TYPES)]
    actors = defaultdict(set)
    for s, d in zip(succ_e.src.values, succ_e.dst.values):
        a, b = issue.get(s, pd.NaT), issue.get(d, pd.NaT)
        if pd.notna(a) and (pd.isna(b) or a >= b):
            actors[d].add(s)

    def successors(d):
        seen, frontier = set(), {d}
        for _ in range(args.depth):
            nxt = set()
            for x in frontier:
                nxt |= actors.get(x, set())
            nxt -= seen | {d}
            seen |= nxt
            frontier = nxt
        return {s for s in seen if s in has_arts}

    linked = links[links.status == "matched"].copy()
    linked["succ_docs"] = linked.doc_id.map(lambda d: successors(d))
    need = linked[linked.succ_docs.map(len) > 0].reset_index(drop=True)
    succ_docs = set().union(*need.succ_docs) if len(need) else set()
    print(f"[{time.time()-t0:.0f}s] {len(need)} passages in {need.doc_id.nunique()} documents have "
          f"{len(succ_docs)} successor documents", flush=True)

    # ---------------- chunk successor articles
    sa = articles[articles.doc_id.isin(succ_docs)].reset_index(drop=True)
    rows = []
    for aid, doc_id, art_no, heading, text in sa[["aid", "doc_id", "art_no", "heading", "text"]].itertuples(index=False):
        for k, ch in enumerate(chunk_article(text, args.max_words)):
            rows.append((f"{aid}~{k}", aid, doc_id, art_no, heading, ch))
    chunks = pd.DataFrame(rows, columns=["chunk_id", "aid", "doc_id", "art_no", "heading", "text"])
    print(f"[{time.time()-t0:.0f}s] {len(sa)} successor articles -> {len(chunks)} chunks", flush=True)

    # ---------------- encode
    model = SentenceTransformer(args.model, revision=args.revision, device="cuda")
    model.half()
    model.max_seq_length = 512
    text_of = corpus.set_index("cid").text
    p_emb = model.encode([text_of[c] for c in need.cid], batch_size=args.batch_size, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=True)
    c_emb = model.encode(chunks.text.tolist(), batch_size=args.batch_size, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=True)
    print(f"[{time.time()-t0:.0f}s] encoded", flush=True)

    # ---------------- align
    chunk_rows_by_doc = chunks.groupby("doc_id").indices
    explicit = set(zip(xref.src_doc.values, xref.dst_aid.values))
    res = []
    for i, r in enumerate(need.itertuples(index=False)):
        idx = np.concatenate([chunk_rows_by_doc[s] for s in r.succ_docs if s in chunk_rows_by_doc]) \
            if any(s in chunk_rows_by_doc for s in r.succ_docs) else np.array([], dtype=int)
        if len(idx) == 0:
            continue
        sims = (c_emb[idx] @ p_emb[i]).float().cpu().numpy()
        df = pd.DataFrame({"aid": chunks.aid.values[idx], "doc_id": chunks.doc_id.values[idx], "sim": sims})
        best = df.groupby(["aid", "doc_id"], as_index=False).sim.max().nlargest(args.topn, "sim")
        for rank, b in enumerate(best.itertuples(index=False)):
            res.append((int(r.cid), r.aid, r.doc_id, b.aid, b.doc_id, round(float(b.sim), 4), rank,
                        (b.doc_id, r.aid) in explicit))
    succ = pd.DataFrame(res, columns=["cid", "aid", "doc_id", "succ_aid", "succ_doc", "sim", "rank", "explicit"])
    succ.to_parquet(out / "successors.parquet", index=False)

    # ---------------- extended corpus with metadata
    dmeta = docs[DOC_COLS].rename(columns={"id": "doc_id"})
    base = corpus.merge(links[["cid", "aid", "doc_id"]], on="cid", how="left")
    base["source"] = "qa"
    in_qa = set(links.aid.dropna())
    new = chunks[~chunks.aid.isin(in_qa)].reset_index(drop=True)
    new = new.assign(cid=np.arange(2_000_000, 2_000_000 + len(new)), source="successor")
    ext = pd.concat([base[["cid", "text", "aid", "doc_id", "source"]],
                     new[["cid", "text", "aid", "doc_id", "source", "chunk_id"]]], ignore_index=True)
    ext = ext.merge(dmeta, on="doc_id", how="left")
    ext.to_parquet(out / "corpus_ext.parquet", index=False)
    docs[DOC_COLS + ["ngay_ban_hanh", "co_quan_ban_hanh", "linh_vuc"]].to_parquet(out / "docs_temporal.parquet", index=False)

    top1 = succ[succ["rank"] == 0]
    stats = {
        "passages_with_successor_docs": len(need),
        "successor_docs": len(succ_docs),
        "successor_articles": len(sa),
        "successor_chunks": len(chunks),
        "extended_corpus": len(ext),
        "extended_corpus_new_rows": len(new),
        "top1_sim_quantiles": {str(q): round(float(top1.sim.quantile(q)), 4) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
        "explicit_any": int(succ.groupby("cid").explicit.any().sum()),
        "top1_explicit": int(top1.explicit.sum()),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "temporal_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
