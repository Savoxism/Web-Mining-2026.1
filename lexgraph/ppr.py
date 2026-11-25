"""Personalized PageRank over the legal document graph, seeded on the candidates of one question.

PPR(s) = alpha * s + (1 - alpha) * P^T PPR(s), with P the row-normalised (symmetrised, type-weighted) adjacency matrix.
Computed for a batch of questions at once (sparse matrix x dense matrix), 30 power iterations.

support(d) = PPR(d) - alpha * s(d): the mass that reaches document d through the graph from the *other* seeds, so a
candidate is not rewarded for being a seed itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

DEFAULT_TYPE_WEIGHTS = {"guides": 1.0, "based_on": 0.5, "cites": 0.5, "amends": 1.0, "replaces": 1.0,
                        "repeals": 0.5}


class DocGraph:
    def __init__(self, edges: pd.DataFrame, doc_ids: list[str], type_weights: dict | None = None):
        tw = type_weights or DEFAULT_TYPE_WEIGHTS
        self.index = {d: i for i, d in enumerate(doc_ids)}
        n = len(doc_ids)
        e = edges[edges.type.isin(tw) & edges.src.isin(self.index) & edges.dst.isin(self.index)]
        r = e.src.map(self.index).to_numpy()
        c = e.dst.map(self.index).to_numpy()
        w = e.type.map(tw).to_numpy(dtype=float)
        A = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n)).tocsr()
        A.sum_duplicates()
        deg = np.asarray(A.sum(axis=1)).ravel()
        inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
        self.PT = (sp.diags(inv) @ A).T.tocsr()  # transpose of the row-stochastic matrix
        self.n = n
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        coo = self.PT.tocoo()
        self.PT_t = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]), coo.data.astype(np.float32), (n, n),
                                            device=self.device).coalesce().to_sparse_csr()

    def ppr(self, seeds: list[dict[int, float]], alpha: float = 0.3, iters: int = 30) -> np.ndarray:
        """seeds: per question, {doc index: weight}. Returns an (n, B) array of PPR scores."""
        B = len(seeds)
        S = np.zeros((self.n, B), dtype=np.float32)
        for j, sd in enumerate(seeds):
            tot = sum(sd.values()) or 1.0
            for i, wgt in sd.items():
                S[i, j] += wgt / tot
        St = torch.from_numpy(S).to(self.device)
        R = St.clone()
        for _ in range(iters):
            R = alpha * St + (1 - alpha) * (self.PT_t @ R)
        return R.cpu().numpy(), S
