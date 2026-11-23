"""Clause-level amendment extraction ("git diff for laws").

Amending documents state their changes as operations, e.g.
    4. Sửa đổi, bổ sung một số điểm, khoản của Điều 6 như sau:
    a) Sửa đổi điểm a khoản 3 Điều 6 như sau:
    "a) Chuyển hướng không giảm tốc độ ...";
    c) Sửa đổi khoản 5 Điều 6 như sau:
    "5. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng ...";
    5. Bãi bỏ điểm c khoản 2 Điều 9.
We parse every document that amends another (typed edge "amends"), extract (operation, target article,
clause, points, new text), resolve the target document from the article heading ("... của Nghị định số
100/2019/NĐ-CP") or the amends edge, and check that the target article / clause exists in the KG.

Usage:
    .venv/bin/python scripts/kg/build_amendments.py --vbpl data/vbpl/data --kg data/kg --out data/kg/amendments.parquet
"""

import argparse
import json
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kg import DOCNUM_RE, html_to_lines  # noqa: E402

OP_RE = re.compile(
    r"^(?:\d+[a-zđ]?\.|[a-zđ]\))?\s*(Sửa đổi, bổ sung|Sửa đổi|Bổ sung|Bãi bỏ|Thay thế|Thay cụm từ|Thay)\b(.*)$")
ART_REF = re.compile(r"Điều\s+(\d+[a-zđ]?)")
CLAUSE_REF = re.compile(r"khoản\s+(\d+[a-zđ]?)")
POINT_REF = re.compile(r"điểm\s+([a-zđ]\d?)\b")
QUOTE_OPEN = ("\"", "“", "'", "‘")
END_Q = re.compile(r"[\"”']\s*[\.;,]?\s*$")
HEAD_RE = re.compile(r"^Điều\s+(\d+[a-zđ]?)\s*[\.:]?\s*(.*)$")


def parse_doc(args):
    doc_id, html, targets = args
    lines = html_to_lines(html)
    ops, cur_head_target, i = [], None, 0
    in_quote = False
    while i < len(lines):
        line = lines[i]
        if not in_quote:
            h = HEAD_RE.match(line)
            if h:  # article of the amending document: which document does it amend?
                nums = DOCNUM_RE.findall(h.group(2))
                cur_head_target = nums[0] if nums else None
                i += 1
                continue
            m = OP_RE.match(line)
            if m and ("Điều" in line or "khoản" in line or "điểm" in line):
                op, spec = m.group(1), m.group(2)
                arts = ART_REF.findall(spec)
                clause = CLAUSE_REF.findall(spec)
                points = POINT_REF.findall(spec)
                nums = DOCNUM_RE.findall(spec)
                new_lines = []
                j = i + 1
                container = spec.rstrip().endswith("như sau:") and (j < len(lines) and not lines[j].startswith(QUOTE_OPEN))
                if spec.rstrip().endswith(":") and not container:
                    # collect the quoted new text
                    while j < len(lines):
                        t = lines[j]
                        if not new_lines and not t.startswith(QUOTE_OPEN):
                            break
                        new_lines.append(t)
                        j += 1
                        if END_Q.search(t) or len(new_lines) > 200:
                            break
                if arts and not container:
                    ops.append({
                        "amend_doc": doc_id, "op": op, "spec": spec.strip()[:300],
                        "target_num": nums[0] if nums else cur_head_target,
                        "target_art": arts[-1], "target_clause": clause[-1] if clause else None,
                        "target_points": ",".join(points) if points else None,
                        "new_text": "\n".join(new_lines).strip("\"“”' ;.")[:6000] if new_lines else None,
                    })
                i = j if new_lines else i + 1
                continue
        i += 1
    return ops


def clause_text(article: str, clause: str | None, points: str | None) -> str | None:
    """Return the old text of a clause (and point) from an article's full text."""
    if not article or not clause:
        return None
    lines = article.split("\n")
    out, inside = [], False
    for l in lines:
        m = re.match(r"^(\d+[a-zđ]?)\.\s", l)
        if m:
            if inside:
                break
            inside = m.group(1) == clause
        if inside:
            out.append(l)
    if not out:
        return None
    if points:
        p = points.split(",")[0]
        pl, keep = [], False
        for l in out[1:]:
            m = re.match(r"^([a-zđ]\d?)\)\s", l)
            if m:
                if keep:
                    break
                keep = m.group(1) == p
            if keep:
                pl.append(l)
        return "\n".join(pl) or None
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vbpl", default="data/vbpl/data")
    ap.add_argument("--kg", default="data/kg")
    ap.add_argument("--out", default="data/kg/amendments.parquet")
    ap.add_argument("--workers", type=int, default=36)
    args = ap.parse_args()
    kg = Path(args.kg)
    docs = pd.read_parquet(kg / "docs.parquet")
    edges = pd.read_parquet(kg / "edges.parquet")
    am = edges[edges.type == "amends"]
    targets_of = am.groupby("src").dst.apply(list).to_dict()
    num2id = (docs.sort_values("ngay_ban_hanh", ascending=False).dropna(subset=["so_ky_hieu"])
              .drop_duplicates("so_ky_hieu").set_index("so_ky_hieu").id.to_dict())
    amend_docs = set(am.src)

    pf = pq.ParquetFile(f"{args.vbpl}/content.parquet")

    def jobs():
        for b in pf.iter_batches(batch_size=256, columns=["id", "content_html"]):
            for i, h in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
                if i in amend_docs and h:
                    yield i, h, targets_of.get(i, [])

    rows = []
    with Pool(args.workers) as pool:
        for ops in pool.imap_unordered(parse_doc, jobs(), chunksize=8):
            rows.extend(ops)
    ops = pd.DataFrame(rows)

    # resolve target document: explicit number -> id; else the single amends target
    def resolve(r):
        if r.target_num and r.target_num in num2id:
            return num2id[r.target_num]
        t = targets_of.get(r.amend_doc, [])
        return t[0] if len(t) == 1 else None
    ops["target_doc"] = ops.apply(resolve, axis=1)
    ops["target_aid"] = ops.target_doc + "#" + ops.target_art

    arts = pd.read_parquet(kg / "articles.parquet", columns=["aid", "text"]).set_index("aid").text
    ops["target_article_exists"] = ops.target_aid.isin(arts.index)
    ops["old_text"] = [clause_text(arts.get(a), c, p) if e else None
                       for a, c, p, e in zip(ops.target_aid, ops.target_clause, ops.target_points, ops.target_article_exists)]
    dm = docs.set_index("id")
    ops["amend_num"] = ops.amend_doc.map(dm.so_ky_hieu)
    ops["amend_from"] = ops.amend_doc.map(pd.to_datetime(dm.ngay_co_hieu_luc, errors="coerce"))
    ops.to_parquet(args.out, index=False)

    stats = {
        "amending_documents": len(amend_docs),
        "documents_with_ops": int(ops.amend_doc.nunique()),
        "operations": len(ops),
        "op_types": ops.op.value_counts().to_dict(),
        "target_resolved": round(float(ops.target_doc.notna().mean()), 4),
        "target_article_found": round(float(ops.target_article_exists.mean()), 4),
        "clause_level_ops": int(ops.target_clause.notna().sum()),
        "old_clause_found": round(float(ops[ops.target_clause.notna() & ops.target_article_exists].old_text.notna().mean()), 4),
        "ops_with_new_text": round(float(ops.new_text.notna().mean()), 4),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    ex = ops[(ops.target_doc == "140152") & (ops.target_art == "6")].head(8)
    print(ex[["amend_num", "op", "target_clause", "target_points", "spec"]].to_string())


if __name__ == "__main__":
    main()
