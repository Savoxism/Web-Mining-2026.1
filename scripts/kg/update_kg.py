"""Incremental knowledge-graph update from newly crawled vbpl.vn documents.

Input: data/crawl/pages_new.parquet (crawl_vbpl.py) and the current KG / temporal layer. Output: a *live* copy
(data/kg_live/, data/kg_live/temporal/) that the pipeline and the demo load instead of the snapshot:
  1. new documents with validity intervals (effFrom / effTo / status from the page);
  2. new articles (HTML -> Điều) and clause-level amendments (same parser as build_amendments.py);
  3. new typed edges extracted from the text (extract_relations.py);
  4. older documents replaced or repealed by a new one get valid_to = the new document's effective date;
  5. QA passages of affected documents are aligned with the new articles (bi-encoder, top-3, cosine >= tau or an
     explicit clause-level amendment), and every new article is chunked and added to the corpus;
  6. corpus embeddings are extended (only the new chunks are encoded).

Usage (GPU): .venv/bin/python scripts/kg/update_kg.py --pages data/crawl/pages_new.parquet --model <bi-encoder>
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_amendments import clause_text, parse_doc  # noqa: E402
from build_kg import EXCLUDED_TYPES, html_to_lines, split_articles  # noqa: E402
from extract_relations import Resolver, extract  # noqa: E402
from lexgraph.temporal import chunk_article  # noqa: E402

DOC_COLS = ["id", "so_ky_hieu", "title", "loai_van_ban", "tinh_trang_hieu_luc", "valid_from", "valid_to",
            "pagerank", "authority", "community"]


def to_date(x):
    return pd.to_datetime(x, errors="coerce")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="data/crawl/pages_new.parquet")
    ap.add_argument("--kg", default="data/kg")
    ap.add_argument("--out", default="data/kg_live")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tau", type=float, default=0.80)
    args = ap.parse_args()
    t0 = time.time()
    kg, out = Path(args.kg), Path(args.out)
    (out / "temporal").mkdir(parents=True, exist_ok=True)
    stats = {}

    pages = pd.read_parquet(args.pages)
    stats["crawled_pages"] = len(pages)
    # same selection as the snapshot (build_kg.py): no translations, consolidated texts or non-normative types
    pages = pages[(pages.content_html.str.len() > 0) & (~pages.is_translation.fillna(False).astype(bool))
                  & (~pages.loai_van_ban.isin(EXCLUDED_TYPES))]
    stats["new_documents"] = len(pages)

    # ---------------- documents
    docs_t = pd.read_parquet(kg / "temporal" / "docs_temporal.parquet")
    new_docs = pd.DataFrame({
        "id": pages.id, "so_ky_hieu": pages.so_ky_hieu, "title": pages.title, "loai_van_ban": pages.loai_van_ban,
        "tinh_trang_hieu_luc": pages.tinh_trang_hieu_luc, "valid_from": to_date(pages.ngay_co_hieu_luc).fillna(to_date(pages.ngay_ban_hanh)),
        "valid_to": to_date(pages.ngay_het_hieu_luc), "pagerank": np.nan, "authority": np.nan, "community": -1,
        "ngay_ban_hanh": to_date(pages.ngay_ban_hanh), "co_quan_ban_hanh": pages.co_quan_ban_hanh, "linh_vuc": pages.linh_vuc})
    docs_live = pd.concat([docs_t[~docs_t.id.isin(new_docs.id)], new_docs], ignore_index=True)

    # ---------------- articles, amendments, relations
    arts = []
    for r in pages.itertuples(index=False):
        arts.extend(split_articles(r.id, html_to_lines(r.content_html)))
    arts = pd.DataFrame(arts)
    arts["aid"] = arts.doc_id + "#" + arts.art_no
    arts = arts.drop_duplicates("aid")
    stats["new_articles"] = len(arts)

    num2doc = (docs_live.sort_values("ngay_ban_hanh", ascending=False).dropna(subset=["so_ky_hieu"])
               .drop_duplicates("so_ky_hieu").set_index("so_ky_hieu").id.to_dict())
    res = Resolver(docs_live[["id", "so_ky_hieu", "ngay_ban_hanh", "loai_van_ban", "title"]])
    rel = []
    for r in pages.itertuples(index=False):
        issued = pd.to_datetime(r.ngay_ban_hanh, errors="coerce")
        for dst, ty in extract(r.title, r.content_html, r.id, res, None if pd.isna(issued) else issued.date()):
            rel.append({"src": r.id, "dst": dst, "type": ty, "label": "extracted"})
    rel = pd.DataFrame(rel, columns=["src", "dst", "type", "label"])
    edges = pd.read_parquet(kg / "edges.parquet")
    edges_live = pd.concat([edges, rel], ignore_index=True).drop_duplicates(["src", "dst", "type"])
    stats["new_edges"] = len(rel)
    stats["new_edge_types"] = rel.type.value_counts().to_dict() if len(rel) else {}

    ops = []
    for r in pages.itertuples(index=False):
        ops.extend(parse_doc((r.id, r.content_html, [])))
    ops = pd.DataFrame(ops)
    if len(ops):
        ops["target_doc"] = ops.target_num.map(num2doc)
        ops["target_aid"] = ops.target_doc + "#" + ops.target_art
        old_arts = pd.read_parquet(kg / "articles.parquet", columns=["aid", "text"]).set_index("aid").text
        ops["target_article_exists"] = ops.target_aid.isin(old_arts.index)
        ops["old_text"] = [clause_text(old_arts.get(a), c, p) if e else None for a, c, p, e in
                           zip(ops.target_aid, ops.target_clause, ops.target_points, ops.target_article_exists)]
        dm = new_docs.set_index("id")
        ops["amend_num"] = ops.amend_doc.map(dm.so_ky_hieu)
        ops["amend_from"] = ops.amend_doc.map(dm.valid_from)
    amend_all = pd.concat([pd.read_parquet(kg / "amendments.parquet"), ops], ignore_index=True) if len(ops) else \
        pd.read_parquet(kg / "amendments.parquet")
    stats["new_amendment_ops"] = len(ops)

    # ---------------- validity of older documents replaced / repealed by new ones
    eff = new_docs.set_index("id").valid_from
    upd = 0
    if len(rel):
        for r in rel[rel.type.isin(["replaces", "repeals"])].itertuples(index=False):
            f = eff.get(r.src)
            m = docs_live.id == r.dst
            if pd.notna(f) and m.any():
                cur = docs_live.loc[m, "valid_to"].iloc[0]
                if pd.isna(cur) or f < cur:
                    docs_live.loc[m, "valid_to"] = f
                    upd += 1
    stats["older_documents_closed"] = upd

    # ---------------- corpus: chunks of all new articles
    corpus = pd.read_parquet(kg / "temporal" / "corpus_ext.parquet")
    rows = []
    for a in arts.itertuples(index=False):
        for k, ch in enumerate(chunk_article(a.text, 300)):
            rows.append((f"{a.aid}~{k}", a.aid, a.doc_id, ch))
    new_chunks = pd.DataFrame(rows, columns=["chunk_id", "aid", "doc_id", "text"])
    start = int(corpus.cid.max()) + 1
    new_chunks["cid"] = np.arange(start, start + len(new_chunks))
    new_chunks["source"] = "crawl"
    dmeta = docs_live[DOC_COLS].rename(columns={"id": "doc_id"})
    new_chunks = new_chunks.merge(dmeta, on="doc_id", how="left")
    corpus_live = pd.concat([corpus, new_chunks[corpus.columns.intersection(new_chunks.columns)]], ignore_index=True)
    # refresh validity columns of all rows (older documents may have been closed)
    vt = docs_live.set_index("id").valid_to
    corpus_live["valid_to"] = corpus_live.doc_id.map(vt).where(corpus_live.doc_id.isin(vt.index), corpus_live.valid_to)
    stats["corpus_rows_closed"] = int((corpus_live.valid_to.notna() & corpus.valid_to.reindex(corpus_live.index).isna()
                                       & (corpus_live.index < len(corpus))).sum())
    stats["new_chunks"] = len(new_chunks)

    # ---------------- align QA passages of affected documents with the new articles
    model = SentenceTransformer(args.model, device="cuda")
    model.half()
    model.max_seq_length = 512
    succ_old = pd.read_parquet(kg / "temporal" / "successors.parquet")
    affected = rel[rel.type.isin(["replaces", "repeals", "partially_repeals", "amends"])] if len(rel) else pd.DataFrame(columns=["src", "dst"])
    qa = corpus[(corpus.source == "qa") & corpus.doc_id.isin(set(affected.dst))]
    new_succ = []
    if len(qa) and len(new_chunks):
        q_emb = model.encode(qa.text.tolist(), batch_size=128, normalize_embeddings=True, convert_to_tensor=True)
        c_emb = model.encode(new_chunks.text.tolist(), batch_size=128, normalize_embeddings=True, convert_to_tensor=True)
        by_doc = new_chunks.groupby("doc_id").indices
        actors = affected.groupby("dst").src.apply(set).to_dict()
        op_pairs = set(zip(ops.amend_doc, ops.target_aid)) if len(ops) else set()
        for i, r in enumerate(qa.itertuples(index=False)):
            idx = [j for s in actors.get(r.doc_id, ()) for j in by_doc.get(s, [])]
            if not idx:
                continue
            sims = (c_emb[idx] @ q_emb[i]).float().cpu().numpy()
            df = pd.DataFrame({"aid": new_chunks.aid.values[idx], "doc": new_chunks.doc_id.values[idx], "sim": sims})
            best = df.groupby(["aid", "doc"], as_index=False).sim.max().nlargest(3, "sim")
            for rank, b in enumerate(best.itertuples(index=False)):
                new_succ.append((int(r.cid), r.aid, r.doc_id, b.aid, b.doc, round(float(b.sim), 4), rank,
                                 (b.doc, r.aid) in op_pairs))
    new_succ = pd.DataFrame(new_succ, columns=["cid", "aid", "doc_id", "succ_aid", "succ_doc", "sim", "rank", "explicit"])
    succ_live = pd.concat([succ_old, new_succ], ignore_index=True)
    stats["affected_qa_passages"] = int(len(qa))
    stats["new_alignments_accepted"] = int(((new_succ.sim >= args.tau) | new_succ.explicit).sum()) if len(new_succ) else 0

    # ---------------- write the live KG
    docs_live.to_parquet(out / "temporal" / "docs_temporal.parquet", index=False)
    corpus_live.to_parquet(out / "temporal" / "corpus_ext.parquet", index=False)
    succ_live.to_parquet(out / "temporal" / "successors.parquet", index=False)
    v2 = kg / "temporal" / "successors_v2.parquet"
    if v2.exists():  # new pairs have no classifier score yet: accepted with the v1 rule
        ns = new_succ.assign(prob=np.nan, accept=(new_succ.sim >= args.tau) | new_succ.explicit)
        pd.concat([pd.read_parquet(v2), ns], ignore_index=True).to_parquet(out / "temporal" / "successors_v2.parquet", index=False)
    edges_live.to_parquet(out / "edges.parquet", index=False)
    amend_all.to_parquet(out / "amendments.parquet", index=False)
    pd.concat([pd.read_parquet(kg / "articles.parquet", columns=["aid", "doc_id", "art_no", "heading", "text"]),
               arts[["aid", "doc_id", "art_no", "heading", "text"]]], ignore_index=True).to_parquet(out / "articles.parquet", index=False)
    shutil.copy(kg / "kg_stats.json", out / "kg_stats.json")

    # extend the embedding cache (only new chunks are encoded)
    import hashlib
    old_key = hashlib.md5(f"{args.model}|None|{len(corpus)}|512".encode()).hexdigest()[:12]
    new_key = hashlib.md5(f"{args.model}|None|{len(corpus_live)}|512".encode()).hexdigest()[:12]
    cache = Path("outputs/cache")
    if (cache / f"emb_{old_key}.pt").exists() and len(new_chunks):
        old = torch.load(cache / f"emb_{old_key}.pt", map_location="cuda")
        add = model.encode(new_chunks.text.tolist(), batch_size=128, normalize_embeddings=True, convert_to_tensor=True)
        torch.save(torch.cat([old, add.to(old.dtype)]), cache / f"emb_{new_key}.pt")
        stats["embedding_cache"] = f"extended {old.shape[0]} -> {old.shape[0] + add.shape[0]}"
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "update_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
