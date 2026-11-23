"""Validity intervals of legal documents and article chunking."""

import re

import numpy as np
import pandas as pd

EXPIRED_STATUSES = {"Hết hiệu lực toàn bộ", "Ngưng hiệu lực", "Không còn phù hợp"}
SUCCESSION_TYPES = {"replaces", "repeals", "partially_repeals", "amends"}
FULL_EXPIRY_TYPES = {"replaces", "repeals"}


def doc_intervals(docs: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Add valid_from / valid_to (Timestamp, NaT = open) to the documents table.

    valid_to comes from the portal's expiry date; for documents marked as expired without a
    date, it is the earliest effective date of a document that repeals or replaces them.
    """
    d = docs.copy()
    d["valid_from"] = d.ngay_co_hieu_luc.fillna(d.ngay_ban_hanh)
    d["valid_to"] = d.ngay_het_hieu_luc
    vf = d.set_index("id").valid_from
    ex = edges[edges.type.isin(FULL_EXPIRY_TYPES)].copy()
    ex["actor_from"] = ex.src.map(vf)
    first_repeal = ex.groupby("dst").actor_from.min()
    missing = d.valid_to.isna() & d.tinh_trang_hieu_luc.isin(EXPIRED_STATUSES)
    d.loc[missing, "valid_to"] = d.loc[missing, "id"].map(first_repeal)
    still = d.valid_to.isna() & d.tinh_trang_hieu_luc.isin(EXPIRED_STATUSES)
    d.loc[still, "valid_to"] = d.loc[still, "valid_from"]  # expired, date unknown
    return d


def in_force(valid_from, valid_to, t: pd.Timestamp) -> np.ndarray:
    vf = pd.to_datetime(pd.Series(valid_from))
    vt = pd.to_datetime(pd.Series(valid_to))
    ok = (vf.isna() | (vf <= t)) & (vt.isna() | (t < vt))
    return ok.to_numpy()


CLAUSE_RE = re.compile(r"^(\d+[a-zđ]?\.|[a-zđ]\)|-)\s")


def chunk_article(text: str, max_words: int = 300) -> list[str]:
    """Split a long article into chunks of whole clauses; each chunk repeats the article heading."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return []
    head, body = lines[0], lines[1:]
    if len(text.split()) <= max_words or not body:
        return [text]
    chunks, cur, n = [], [], 0
    for line in body:
        w = len(line.split())
        if cur and n + w > max_words and (CLAUSE_RE.match(line) or n > 2 * max_words):
            chunks.append(cur)
            cur, n = [], 0
        cur.append(line)
        n += w
    if cur:
        chunks.append(cur)
    return [head + "\n" + "\n".join(c) for c in chunks]
