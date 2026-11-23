"""Retrieval pipeline: bi-encoder -> (successor injection) -> cross-encoder -> graph + time re-ranking.

    s(q, d) = s_CE(q, d) + beta * z(log PR(d)) - gamma * 1[d not in force at t_q] + mu * prop(d)

prop(d) is the mean of the top-3 sigmoid cross-encoder scores of other candidates that share d's
document or are linked to it by a guides / based_on / cites edge.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lexgraph.query import expand_query
from lexgraph.temporal import chunk_article, in_force

GRAPH_TYPES = {"guides", "based_on", "cites"}


@dataclass
class Weights:
    beta: float = 0.0
    gamma: float = 0.0
    mu: float = 0.0
    tau: float = 0.80
    use_ce: bool = True
    inject: bool = True
    static: bool = False   # retrieve only from the original QA corpus (no successor articles)
    expand: bool = True    # query normalisation with the legal glossary (demo; off in the benchmark evaluation)


class LexGraph:
    def __init__(self, bi_model: str, ce_model: str | None = None, temporal_dir: str = "data/kg/temporal",
                 kg_dir: str = "data/kg", device: str = "cuda", bi_revision: str | None = None,
                 ce_revision: str | None = None, cache_dir: str = "outputs/cache", max_seq_len: int = 512):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.device = device
        td, kg = Path(temporal_dir), Path(kg_dir)
        self.corpus = pd.read_parquet(td / "corpus_ext.parquet").reset_index(drop=True)
        c = self.corpus
        self.cids = c.cid.to_numpy()
        self.row_of = {int(x): i for i, x in enumerate(self.cids)}
        self.valid_from = pd.to_datetime(c.valid_from).to_numpy()
        self.valid_to = pd.to_datetime(c.valid_to).to_numpy()
        self.doc_ids = c.doc_id.fillna("").to_numpy()
        pr = c.pagerank.astype(float)
        lpr = np.log(pr.where(pr > 0))
        self.z_pr = ((lpr - lpr.mean()) / lpr.std()).fillna(0.0).to_numpy()

        # rows of each article (QA passage or successor chunks)
        self.rows_of_aid = defaultdict(list)
        for i, a in enumerate(c.aid.to_numpy()):
            if isinstance(a, str):
                self.rows_of_aid[a].append(i)

        # successor candidates of QA passages
        succ = pd.read_parquet(td / "successors.parquet")
        docs = pd.read_parquet(td / "docs_temporal.parquet").set_index("id")
        self.docs = docs
        succ["succ_from"] = succ.succ_doc.map(docs.valid_from)
        succ["succ_to"] = succ.succ_doc.map(docs.valid_to)
        self.succ_by_row = defaultdict(list)
        for r in succ.itertuples(index=False):
            i = self.row_of.get(int(r.cid))
            if i is not None:
                self.succ_by_row[i].append(r)

        # document adjacency for graph propagation
        edges = pd.read_parquet(kg / "edges.parquet")
        present = set(self.doc_ids) - {""}
        e = edges[edges.type.isin(GRAPH_TYPES) & edges.src.isin(present) & edges.dst.isin(present)]
        self.adj = defaultdict(set)
        for s, d in zip(e.src.values, e.dst.values):
            self.adj[s].add(d)
            self.adj[d].add(s)
        # direct succession edge types between (successor doc, old doc), for edge-type ablations
        se = edges[edges.type.isin({"amends", "replaces", "repeals", "partially_repeals"}) & edges.dst.isin(present)]
        self.pair_types = defaultdict(set)
        for s, d, ty in zip(se.src.values, se.dst.values, se.type.values):
            self.pair_types[(s, d)].add(ty)

        # full article texts for QA passages (the QA corpus sometimes elides clauses with "...")
        self.full_text = {}
        art_path = kg / "articles.parquet"
        if art_path.exists():
            aids = sorted(set(c.aid[c.source == "qa"].dropna()))
            arts = pd.read_parquet(art_path, columns=["aid", "text"], filters=[("aid", "in", aids)])
            self.full_text = dict(zip(arts.aid, arts.text))

        self.bi = SentenceTransformer(bi_model, revision=bi_revision, device=device)
        self.bi.max_seq_length = max_seq_len
        if device == "cuda":
            self.bi.half()
        self.ce = None
        if ce_model:
            self.ce = CrossEncoder(ce_model, revision=ce_revision, device=device, max_length=512)
            if device == "cuda":
                self.ce.model.half()
        key = hashlib.md5(f"{bi_model}|{bi_revision}|{len(c)}|{max_seq_len}".encode()).hexdigest()[:12]
        cache = Path(cache_dir) / f"emb_{key}.pt"
        if cache.exists():
            self.emb = torch.load(cache, map_location=device)
        else:
            self.emb = self.bi.encode(c.text.tolist(), batch_size=256, normalize_embeddings=True,
                                      convert_to_tensor=True, show_progress_bar=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.emb, cache)

    # ------------------------------------------------------------------ temporal status
    def best_successor(self, i: int, t: pd.Timestamp, tau: float, use_explicit: bool = True,
                       allowed_types: set | None = None):
        """Best successor article in force at t (explicit reference first, then similarity).

        use_explicit=False ignores extracted cross-references (similarity only);
        allowed_types keeps only successors linked to the old document by a direct edge of those types.
        """
        best, best_key = None, None
        for r in self.succ_by_row.get(i, []):
            explicit = bool(r.explicit) and use_explicit
            if not (explicit or r.sim >= tau):
                continue
            if allowed_types is not None and not (self.pair_types.get((r.succ_doc, r.doc_id), set()) & allowed_types):
                continue
            f, to = r.succ_from, r.succ_to
            if not ((pd.isna(f) or f <= t) and (pd.isna(to) or t < to)):
                continue
            key = (explicit, r.sim)
            if best is None or key > best_key:
                best, best_key = r, key
        return best

    def status(self, rows: np.ndarray, t: pd.Timestamp, tau: float, use_explicit: bool = True,
               allowed_types: set | None = None):
        """Return (stale flags, reasons, successor records) for corpus rows at time t."""
        rows = np.asarray(rows, dtype=int)
        if len(rows) == 0:
            return np.zeros(0, dtype=bool), [], []
        doc_ok = in_force(self.valid_from[rows], self.valid_to[rows], t)
        linked = self.doc_ids[rows] != ""
        stale, reasons, succs = [], [], []
        for k, i in enumerate(rows):
            s = self.best_successor(int(i), t, tau, use_explicit, allowed_types)
            if linked[k] and not doc_ok[k]:
                stale.append(True); reasons.append("document not in force")
            elif s is not None:
                stale.append(True); reasons.append("article superseded")
            else:
                stale.append(False); reasons.append("")
            succs.append(s)
        return np.array(stale, dtype=bool), reasons, succs

    # ------------------------------------------------------------------ retrieval
    def relevant_text(self, q_vec: torch.Tensor, aid: str, fallback: str, n: int = 2, max_words: int = 300) -> str:
        """Most query-relevant clauses of the full article (falls back to the passage text)."""
        full = self.full_text.get(aid) if isinstance(aid, str) else None
        if not full or len(full) <= len(fallback) * 1.05:
            return fallback
        chunks = chunk_article(full, max_words)
        if len(chunks) <= 1:
            return chunks[0] if chunks else fallback
        emb = self.bi.encode(chunks, batch_size=32, normalize_embeddings=True, convert_to_tensor=True)
        sims = (emb @ q_vec).float().cpu().numpy()
        top = sorted(np.argsort(-sims)[:n])
        return "\n…\n".join(chunks[i] for i in top)

    def encode_queries(self, queries: list[str]) -> torch.Tensor:
        return self.bi.encode(queries, batch_size=256, normalize_embeddings=True, convert_to_tensor=True,
                              show_progress_bar=False)

    def retrieve(self, q_emb: torch.Tensor, k: int = 100):
        out_i, out_s = [], []
        for start in range(0, len(q_emb), 1024):
            s = q_emb[start:start + 1024] @ self.emb.T
            ts, ti = torch.topk(s, k=k, dim=1)
            out_i.append(ti.cpu().numpy()); out_s.append(ts.float().cpu().numpy())
        return np.concatenate(out_i), np.concatenate(out_s)

    def candidates(self, q_vec: torch.Tensor, top_rows: np.ndarray, t: pd.Timestamp, tau: float,
                   k_rerank: int, inject: bool, use_explicit: bool = True,
                   allowed_types: set | None = None) -> np.ndarray:
        rows = [int(r) for r in top_rows[:k_rerank]]
        if inject:
            _, _, succs = self.status(np.array(rows), t, tau, use_explicit, allowed_types)
            extra = []
            for s in succs:
                if s is None:
                    continue
                cand = self.rows_of_aid.get(s.succ_aid, [])
                if cand:
                    sims = (self.emb[cand] @ q_vec).float().cpu().numpy()
                    extra.extend(np.array(cand)[np.argsort(-sims)[:2]].tolist())
            seen = set(rows)
            rows += [r for r in extra if not (r in seen or seen.add(r))]
        return np.array(rows)

    def ce_scores(self, query: str, rows: np.ndarray) -> np.ndarray:
        pairs = [(query, self.corpus.text.iat[int(r)]) for r in rows]
        return np.asarray(self.ce.predict(pairs, batch_size=64, show_progress_bar=False,
                                          activation_fn=torch.nn.Identity()), dtype=float)

    def propagation(self, rows: np.ndarray, ce: np.ndarray) -> np.ndarray:
        prob = 1 / (1 + np.exp(-ce))
        docs = self.doc_ids[rows]
        out = np.zeros(len(rows))
        for a in range(len(rows)):
            da = docs[a]
            if not da:
                continue
            nb = [prob[b] for b in range(len(rows)) if b != a and docs[b] and (docs[b] == da or docs[b] in self.adj[da])]
            if nb:
                out[a] = float(np.mean(sorted(nb, reverse=True)[:3]))
        return out

    def combine(self, rows, base, stale, prop, w: Weights) -> np.ndarray:
        return base + w.beta * self.z_pr[rows] - w.gamma * stale.astype(float) + w.mu * prop

    # ------------------------------------------------------------------ single query (demo)
    def search(self, query: str, t: pd.Timestamp, w: Weights, k_retrieve: int = 100, k_rerank: int = 30,
               top: int = 10) -> pd.DataFrame:
        query = expand_query(query) if w.expand else query
        q = self.encode_queries([query])
        ti, ts = self.retrieve(q, k=k_retrieve * (3 if w.static else 1))
        if w.static:
            keep = self.corpus.source.to_numpy()[ti[0]] == "qa"
            ti, ts = ti[:, keep][:, :k_retrieve], ts[:, keep][:, :k_retrieve]
        bi_score = dict(zip(ti[0].tolist(), ts[0].tolist()))
        rows = self.candidates(q[0], ti[0], t, w.tau, k_rerank, w.inject)
        if self.ce is not None and w.use_ce:
            base = self.ce_scores(query, rows)
        else:
            base = (self.emb[rows] @ q[0]).float().cpu().numpy() * 10
        stale, reasons, succs = self.status(rows, t, w.tau)
        prop = self.propagation(rows, base)
        final = self.combine(rows, base, stale, prop, w)
        if w.static:  # a static system knows nothing about validity
            stale = np.zeros(len(rows), dtype=bool)
            reasons, succs = [""] * len(rows), [None] * len(rows)
        res = self.corpus.iloc[rows].copy()
        res["row"] = rows
        res["bi_score"] = [bi_score.get(int(r), float((self.emb[int(r)] @ q[0]).item())) for r in rows]
        res["ce_score"] = base
        res["pagerank_z"] = self.z_pr[rows]
        res["propagation"] = prop
        res["stale"] = stale
        res["stale_reason"] = reasons
        res["successor_aid"] = [s.succ_aid if s is not None else None for s in succs]
        res["successor_doc"] = [s.succ_doc if s is not None else None for s in succs]
        res["successor_sim"] = [s.sim if s is not None else None for s in succs]
        res["successor_explicit"] = [bool(s.explicit) if s is not None else None for s in succs]
        res["final"] = final
        return res.sort_values("final", ascending=False).head(top).reset_index(drop=True)
