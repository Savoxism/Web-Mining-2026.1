"""Version history of an article: clause-level amendments ("git diff for laws") and successors."""

from __future__ import annotations

import difflib
import html
import re
from pathlib import Path

import pandas as pd


class VersionIndex:
    def __init__(self, amendments_path: str = "data/kg/amendments.parquet"):
        p = Path(amendments_path)
        self.ops = pd.read_parquet(p) if p.exists() else pd.DataFrame(
            columns=["target_aid", "amend_doc", "amend_num", "amend_from", "op", "target_clause", "target_points",
                     "old_text", "new_text", "spec"])
        self.by_aid = {a: g.sort_values("amend_from") for a, g in self.ops.groupby("target_aid")}

    def amendments(self, aid: str, t: pd.Timestamp | None = None) -> pd.DataFrame:
        """Clause-level amendments of an article, optionally only those in force at time t."""
        g = self.by_aid.get(aid)
        if g is None:
            return self.ops.iloc[0:0]
        if t is not None:
            g = g[g.amend_from.isna() | (g.amend_from <= t)]
        return g

    def timeline(self, aid: str, lg, t: pd.Timestamp | None = None) -> list[dict]:
        """Events: original article, clause-level amendments, then the successor article (if any)."""
        events = []
        doc = aid.split("#")[0]
        if doc in lg.docs.index:
            d = lg.docs.loc[doc]
            events.append({"date": d.valid_from, "kind": "ban hành / có hiệu lực", "doc": d.so_ky_hieu, "aid": aid})
        for r in self.amendments(aid).itertuples(index=False):
            events.append({"date": r.amend_from, "kind": f"{r.op.lower()}", "doc": r.amend_num, "aid": aid,
                           "clause": r.target_clause, "points": r.target_points, "old": r.old_text, "new": r.new_text,
                           "spec": r.spec})
        rows = lg.rows_of_aid.get(aid, [])
        succ = None
        for i in rows:
            s = lg.best_successor(i, t or pd.Timestamp.today(), 0.80)
            if s is not None:
                succ = s
                break
        if succ is not None and succ.succ_doc in lg.docs.index:
            d = lg.docs.loc[succ.succ_doc]
            events.append({"date": d.valid_from, "kind": "thay thế bởi điều luật mới", "doc": d.so_ky_hieu,
                           "aid": succ.succ_aid, "sim": succ.sim, "explicit": bool(succ.explicit)})
        events.sort(key=lambda e: (pd.Timestamp.min if pd.isna(e["date"]) else e["date"]))
        return events


TOKEN = re.compile(r"\S+|\s+")


def word_diff_html(old: str | None, new: str | None) -> str:
    """Word-level diff: deletions in red strike-through, insertions in green."""
    a, b = TOKEN.findall(old or ""), TOKEN.findall(new or "")
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape("".join(a[i1:i2])))
        if op in ("delete", "replace"):
            out.append('<span style="background:#fde2e1;color:#9b2c2c;text-decoration:line-through">'
                       + html.escape("".join(a[i1:i2])) + "</span>")
        if op in ("insert", "replace"):
            out.append('<span style="background:#dcf3e3;color:#1e6b3a">' + html.escape("".join(b[j1:j2])) + "</span>")
    return '<div style="white-space:pre-wrap;font-size:0.92em;line-height:1.5">' + "".join(out) + "</div>"


MONEY = re.compile(r"(\d{1,3}(?:\.\d{3})+)\s*đồng")


def money_change(old: str | None, new: str | None) -> str | None:
    """Short note when fine amounts changed, e.g. '600.000–1.000.000 đồng → 4.000.000–6.000.000 đồng'."""
    mo, mn = MONEY.findall(old or ""), MONEY.findall(new or "")
    if mo and mn and mo[:2] != mn[:2]:
        return f"{'–'.join(mo[:2])} đồng → {'–'.join(mn[:2])} đồng"
    return None
