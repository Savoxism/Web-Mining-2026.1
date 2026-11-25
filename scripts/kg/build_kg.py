"""Build the legal knowledge graph from the vbpl.vn crawl (th1nhng0/vietnamese-legal-documents).

Steps
  1. Select normative documents and parse their HTML into articles (Điều).
  2. Canonicalise relations into typed, directed edges (actor -> target).
  3. Extract in-text cross-references (document numbers, optionally with an article) and
     evaluate them against the portal's own relation lists.
  4. Link every QA-corpus passage to the article it was taken from.
  5. Link analysis: PageRank, HITS, Louvain communities.

Usage:
    .venv/bin/python scripts/kg/build_kg.py --vbpl data/vbpl/data --corpus data/processed/corpus.parquet --out data/kg
"""

import argparse
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import lxml.html
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# --------------------------------------------------------------------------- config
EXCLUDED_TYPES = {
    "Bản dịch văn bản", "Văn bản hợp nhất", "Văn bản hệ thống hóa", "Công văn", "Chỉ thị",
    "Thông báo", "Điều ước quốc tế", "Hướng dẫn", "Kế hoạch", "Báo cáo", "Chương trình",
}

# Relation label -> (canonical type, direction[, partial]). direction "fwd": doc_id acts on other_doc_id;
# "rev": other_doc_id acts on doc_id. Directions were checked against issue dates
# (actor is newer in >= 98% of edges for every label used here).
# Six edge types. A partial repeal is a "repeals" edge with partial=True (it changes validity of some articles only).
# Rare portal labels are left out: suspension (188 edges), interpretation (85), consolidation (targets are
# consolidated texts, which we exclude), correction, publication, translation and "other related" documents.
RELATION_MAP = {
    "Căn cứ": ("based_on", "fwd"),
    "Văn bản căn cứ": ("based_on", "fwd"),
    "Dẫn chiếu": ("cites", "fwd"),
    "Văn bản dẫn chiếu": ("cites", "fwd"),
    "Quy định chi tiết, hướng dẫn thi hành": ("guides", "fwd"),
    "Hướng dẫn áp dụng": ("guides", "fwd"),
    "Văn bản được HD, QĐ chi tiết": ("guides", "fwd"),
    "Văn bản HD, QĐ chi tiết": ("guides", "rev"),
    "Thay thế": ("replaces", "fwd"),
    "Bãi bỏ": ("repeals", "fwd"),
    "Văn bản hết hiệu lực": ("repeals", "fwd"),
    "Văn bản quy định hết hiệu lực": ("repeals", "rev"),
    "Văn bản bị hết hiệu lực 1 phần": ("repeals", "fwd", True),
    "Văn bản quy định hết hiệu lực 1 phần": ("repeals", "rev", True),
    "Sửa đổi, bổ sung": ("amends", "fwd"),
    "Văn bản được sửa đổi": ("amends", "fwd"),
    "Văn bản được bổ sung": ("amends", "fwd"),
    "Văn bản sửa đổi": ("amends", "rev"),
    "Văn bản bổ sung": ("amends", "rev"),
}
AUTHORITY_TYPES = {"based_on", "cites", "guides"}          # edges used for PageRank / HITS
SUCCESSION_TYPES = {"replaces", "repeals", "amends"}

BLOCK_TAGS = {"p", "div", "li", "tr", "td", "h1", "h2", "h3", "h4", "h5", "h6", "table", "br"}
ART_RE = re.compile(r"^[\"“]?\s*Điều\s+(\d+[a-zđ]?)\s*[\.:\-–]?\s*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^(Chương|CHƯƠNG|Mục|MỤC|Phần|PHẦN)\s+[0-9IVXLC]+\b")
DOCNUM_RE = re.compile(r"(\d{1,4}/\d{4}/[A-ZĐ][A-ZĐa-zđ0-9\-]*[A-ZĐ0-9])")
XREF_ART_RE = re.compile(r"Điều\s+(\d+[a-zđ]?)(?:[^.;\n]{0,80}?)(?:số\s+)?(\d{1,4}/\d{4}/[A-ZĐ][A-ZĐa-zđ0-9\-]*[A-ZĐ0-9])")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    return " ".join(TOKEN_RE.findall(s))


def parse_date(s):
    return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")


# --------------------------------------------------------------------------- 1. parse HTML
def html_to_lines(html: str) -> list[str]:
    try:
        root = lxml.html.fromstring(html)
    except Exception:
        return []
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag.lower() in BLOCK_TAGS:
            el.tail = "\n" + (el.tail or "")
    text = unicodedata.normalize("NFC", root.text_content())
    return [" ".join(l.split()) for l in text.split("\n") if l.strip()]


def split_articles(doc_id: str, lines: list[str]) -> list[dict]:
    arts, cur = [], None
    for line in lines:
        m = ART_RE.match(line)
        if m and not line.startswith(("\"", "“")):
            if cur:
                arts.append(cur)
            cur = {"doc_id": doc_id, "art_no": m.group(1), "heading": m.group(2).strip(), "lines": [line]}
        elif cur is not None:
            if SECTION_RE.match(line):
                arts.append(cur)
                cur = None
            else:
                cur["lines"].append(line)
    if cur:
        arts.append(cur)
    out = []
    for a in arts:
        text = "\n".join(a.pop("lines"))
        a["text"] = text[:20000]
        out.append(a)
    return out


def extract_xrefs(doc_id: str, lines: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (document numbers cited anywhere, [(article_no, document number)] article-level refs)."""
    text = "\n".join(lines)
    nums = DOCNUM_RE.findall(text)
    arts = [(a, n) for a, n in XREF_ART_RE.findall(text)]
    return nums, arts


def parse_one(args):
    doc_id, html = args
    lines = html_to_lines(html)
    arts = split_articles(doc_id, lines)
    nums, art_refs = extract_xrefs(doc_id, lines)
    return doc_id, arts, nums, art_refs, len(lines)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vbpl", default="data/vbpl/data")
    ap.add_argument("--corpus", default="data/processed/corpus.parquet")
    ap.add_argument("--out", default="data/kg")
    ap.add_argument("--workers", type=int, default=36)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    t0 = time.time()

    # ---------------- metadata & document selection
    meta = pd.read_parquet(f"{args.vbpl}/metadata.parquet")
    meta["is_translation"] = meta.id.str.startswith("vbpqta_")
    keep = (~meta.loai_van_ban.isin(EXCLUDED_TYPES)) & (~meta.is_translation)
    docs = meta[keep].copy()
    for c in ["ngay_ban_hanh", "ngay_co_hieu_luc", "ngay_het_hieu_luc"]:
        docs[c] = parse_date(docs[c])
    stats["docs_total"] = len(meta)
    stats["docs_selected"] = len(docs)
    stats["doc_types"] = docs.loai_van_ban.value_counts().head(12).to_dict()
    stats["doc_status"] = docs.tinh_trang_hieu_luc.value_counts().to_dict()
    doc_ids = set(docs.id)
    print(f"[{time.time()-t0:.0f}s] selected {len(docs)} / {len(meta)} documents", flush=True)

    # number -> doc id (prefer normative, non-translation, most recent)
    num2doc = (
        docs.sort_values("ngay_ban_hanh", ascending=False)
        .dropna(subset=["so_ky_hieu"])
        .drop_duplicates("so_ky_hieu")
        .set_index("so_ky_hieu").id.to_dict()
    )

    # ---------------- 2. canonical relations
    rel = pd.read_parquet(f"{args.vbpl}/relationships.parquet")
    rel = rel[rel.relationship.isin(RELATION_MAP)]
    typ = rel.relationship.map(lambda r: RELATION_MAP[r][0])
    fwd = rel.relationship.map(lambda r: RELATION_MAP[r][1] == "fwd")
    edges = pd.DataFrame({
        "src": np.where(fwd, rel.doc_id, rel.other_doc_id),
        "dst": np.where(fwd, rel.other_doc_id, rel.doc_id),
        "type": typ.values,
        "partial": rel.relationship.map(lambda r: len(RELATION_MAP[r]) > 2).values,
        "label": rel.relationship.values,
    })
    edges = edges[edges.src.isin(doc_ids) & edges.dst.isin(doc_ids) & (edges.src != edges.dst)]
    edges = edges.drop_duplicates(["src", "dst", "type", "partial"]).reset_index(drop=True)
    stats["edges"] = len(edges)
    stats["edge_types"] = edges.type.value_counts().to_dict()
    stats["partial_repeal_edges"] = int(edges.partial.sum())
    edges.to_parquet(out / "edges.parquet", index=False)
    print(f"[{time.time()-t0:.0f}s] {len(edges)} canonical edges", flush=True)

    # ---------------- 1. parse content
    pf = pq.ParquetFile(f"{args.vbpl}/content.parquet")
    all_arts, doc_nums, art_refs, n_lines = [], {}, [], {}

    def batches():
        for b in pf.iter_batches(batch_size=512, columns=["id", "content_html"]):
            for i, h in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
                if i in doc_ids and h:
                    yield i, h

    with Pool(args.workers) as pool:
        for doc_id, arts, nums, arefs, nl in tqdm(pool.imap_unordered(parse_one, batches(), chunksize=16),
                                                  total=len(doc_ids), desc="parse", mininterval=30):
            all_arts.extend(arts)
            doc_nums[doc_id] = nums
            art_refs.extend((doc_id, a, n) for a, n in arefs)
            n_lines[doc_id] = nl
    articles = pd.DataFrame(all_arts)
    articles["aid"] = articles.doc_id + "#" + articles.art_no
    articles = articles.drop_duplicates("aid").reset_index(drop=True)
    articles.to_parquet(out / "articles.parquet", index=False)
    stats["docs_with_content"] = len(doc_nums)
    stats["articles"] = len(articles)
    stats["docs_with_articles"] = int(articles.doc_id.nunique())
    print(f"[{time.time()-t0:.0f}s] parsed {len(doc_nums)} docs into {len(articles)} articles", flush=True)

    # ---------------- 3. cross-references (IE) + evaluation against portal relations
    xrefs = []
    for d, nums in doc_nums.items():
        for n in set(nums):
            t = num2doc.get(n)
            if t and t != d:
                xrefs.append((d, t))
    xref_doc = pd.DataFrame(xrefs, columns=["src", "dst"]).drop_duplicates()
    xa = [(d, num2doc.get(n), a) for d, a, n in art_refs if num2doc.get(n) and num2doc.get(n) != d]
    xref_art = pd.DataFrame(xa, columns=["src_doc", "dst_doc", "dst_art_no"]).drop_duplicates()
    xref_art["dst_aid"] = xref_art.dst_doc + "#" + xref_art.dst_art_no
    xref_art["resolved"] = xref_art.dst_aid.isin(set(articles.aid))
    xref_doc.to_parquet(out / "xref_doc_edges.parquet", index=False)
    xref_art.to_parquet(out / "xref_article_edges.parquet", index=False)

    silver = edges[edges.src.isin(doc_nums.keys())][["src", "dst"]].drop_duplicates()
    sv = set(map(tuple, silver.values))
    pr = set(map(tuple, xref_doc.values))
    srcs = set(silver.src) & set(xref_doc.src)
    sv_s = {e for e in sv if e[0] in srcs}
    pr_s = {e for e in pr if e[0] in srcs}
    tp = len(sv_s & pr_s)
    stats["xref_eval"] = {
        "docs_evaluated": len(srcs),
        "predicted_doc_refs": len(pr_s),
        "silver_doc_relations": len(sv_s),
        "precision_vs_portal_relations": round(tp / max(1, len(pr_s)), 4),
        "recall_vs_portal_relations": round(tp / max(1, len(sv_s)), 4),
        "new_refs_not_in_portal": len(pr_s - sv_s),
    }
    stats["xref_article_edges"] = len(xref_art)
    stats["xref_article_resolved"] = round(float(xref_art.resolved.mean()), 4) if len(xref_art) else 0.0
    print(f"[{time.time()-t0:.0f}s] xref eval {stats['xref_eval']}", flush=True)

    # ---------------- 4. link QA passages to articles
    corpus = pd.read_parquet(args.corpus)
    articles["hkey"] = articles.heading.map(norm)
    h2rows = defaultdict(list)
    for i, k in enumerate(articles.hkey.values):
        if k:
            h2rows[k].append(i)
    art_tok = {}

    def toks(i):
        if i not in art_tok:
            art_tok[i] = set(TOKEN_RE.findall(articles.text.iat[i].lower()))
        return art_tok[i]

    issue = docs.set_index("id").ngay_ban_hanh
    links = []
    for cid, text in tqdm(zip(corpus.cid.values, corpus.text.values), total=len(corpus), desc="link", mininterval=30):
        lines = text.strip().strip("\"").split("\n")
        first = lines[0].strip()
        m = ART_RE.match(first)
        art_no, heading = (m.group(1), m.group(2)) if m else (None, first)
        cands = h2rows.get(norm(heading), [])
        if not cands:
            links.append((cid, None, None, 0.0, "no_heading_match", len(cands)))
            continue
        body = set(TOKEN_RE.findall(" ".join(lines[1:]).lower())) or set(TOKEN_RE.findall(heading.lower()))
        best, best_s = None, -1.0
        for i in cands[:50000]:
            s = len(body & toks(i)) / max(1, len(body))
            if art_no and articles.art_no.iat[i] == art_no:
                s += 0.05
            if s > best_s or (abs(s - best_s) < 1e-9 and best is not None and
                              issue.get(articles.doc_id.iat[i], pd.NaT) > issue.get(articles.doc_id.iat[best], pd.NaT)):
                best, best_s = i, s
        ok = best_s >= 0.6
        links.append((cid, articles.aid.iat[best] if ok else None, articles.doc_id.iat[best] if ok else None,
                      round(min(best_s, 1.0), 4), "matched" if ok else "low_overlap", len(cands)))
    links = pd.DataFrame(links, columns=["cid", "aid", "doc_id", "overlap", "status", "n_candidates"])
    links.to_parquet(out / "passage_links.parquet", index=False)
    stats["passage_links"] = links.status.value_counts().to_dict()
    stats["passage_link_rate"] = round(float((links.status == "matched").mean()), 4)
    stats["linked_docs"] = int(links.doc_id.nunique())
    print(f"[{time.time()-t0:.0f}s] linked {stats['passage_link_rate']:.1%} of passages", flush=True)

    # ---------------- 5. link analysis
    G = nx.DiGraph()
    G.add_nodes_from(doc_ids)
    auth = edges[edges.type.isin(AUTHORITY_TYPES)]
    G.add_edges_from(zip(auth.src, auth.dst))
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-8)
    hubs, authorities = nx.hits(G, max_iter=500, tol=1e-8, normalized=True)
    print(f"[{time.time()-t0:.0f}s] pagerank + hits done", flush=True)

    # Louvain on the neighbourhood of documents that the QA corpus uses
    core = set(links.doc_id.dropna())
    nb = set(core)
    for d in core:
        nb.update(G.successors(d))
        nb.update(G.predecessors(d))
    U = G.subgraph(nb).to_undirected()
    comms = nx.community.louvain_communities(U, seed=42, resolution=1.0)
    comm_of = {n: k for k, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c}
    stats["louvain"] = {"nodes": U.number_of_nodes(), "edges": U.number_of_edges(), "communities": len(comms),
                        "modularity": round(nx.community.modularity(U, comms), 4),
                        "largest_sizes": sorted(map(len, comms), reverse=True)[:10]}

    docs["pagerank"] = docs.id.map(pagerank)
    docs["hub"] = docs.id.map(hubs)
    docs["authority"] = docs.id.map(authorities)
    docs["community"] = docs.id.map(comm_of).fillna(-1).astype(int)
    docs["n_articles"] = docs.id.map(articles.doc_id.value_counts()).fillna(0).astype(int)
    docs["linked_passages"] = docs.id.map(links.doc_id.value_counts()).fillna(0).astype(int)
    docs.drop(columns=["is_translation"]).to_parquet(out / "docs.parquet", index=False)

    top = docs.nlargest(15, "pagerank")[["so_ky_hieu", "title", "loai_van_ban", "pagerank"]]
    stats["top_pagerank"] = [{"so_ky_hieu": r.so_ky_hieu, "title": r.title[:90], "type": r.loai_van_ban,
                              "pagerank": round(r.pagerank, 6)} for r in top.itertuples()]
    topa = docs.nlargest(10, "authority")[["so_ky_hieu", "title"]]
    stats["top_hits_authority"] = [f"{r.so_ky_hieu} | {r.title[:80]}" for r in topa.itertuples()]
    # Community labels: most common field (linh_vuc) per large community
    lab = []
    for k in range(min(8, len(comms))):
        members = docs[docs.community == k]
        lab.append({"community": k, "size": len(members),
                    "top_fields": members.linh_vuc.dropna().value_counts().head(3).to_dict()})
    stats["community_fields"] = lab
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "kg_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
