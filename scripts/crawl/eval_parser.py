"""Field-level accuracy of the vbpl.vn page parser on documents that are also in the public crawl (data/vbpl).

Usage: .venv/bin/python scripts/crawl/eval_parser.py --pages data/crawl/pages_known.parquet
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kg"))
from build_kg import html_to_lines  # noqa: E402

TOK = re.compile(r"\w+", re.UNICODE)


def norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().lower() if x is not None and not (isinstance(x, float) and pd.isna(x)) else ""


def day(x):
    d = pd.to_datetime(x, errors="coerce", utc=True)
    return None if pd.isna(d) else (d.tz_convert("Asia/Ho_Chi_Minh").date() if d.tzinfo else d.date())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="data/crawl/pages_known.parquet")
    ap.add_argument("--meta", default="data/vbpl/data/metadata.parquet")
    ap.add_argument("--kg", default="data/kg")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    P = pd.read_parquet(args.pages)
    M = pd.read_parquet(args.meta).drop_duplicates("id").set_index("id")
    P = P[P.id.isin(M.index)]
    res = {"documents": len(P)}
    for f, kind in [("so_ky_hieu", "text"), ("title", "text"), ("loai_van_ban", "text"), ("co_quan_ban_hanh", "text"),
                    ("ngay_ban_hanh", "date"), ("ngay_co_hieu_luc", "date"), ("ngay_het_hieu_luc", "date"),
                    ("tinh_trang_hieu_luc", "text")]:
        if f not in M.columns:
            continue
        ok, n = 0, 0
        for r in P.itertuples(index=False):
            a, b = getattr(r, f), M.at[r.id, f]
            if kind == "date":
                a, b = day(a), day(b)
                if a is None and b is None:
                    continue
                ok += a == b
            else:
                if not norm(b):
                    continue
                ok += norm(a) == norm(b)
            n += 1
        res[f] = {"accuracy": round(ok / max(1, n), 4), "n": n}
    # full text: article split agreement with the KG built from the public crawl
    arts = pd.read_parquet(Path(args.kg) / "articles.parquet", columns=["doc_id", "art_no"])
    kg_n = arts[arts.doc_id.isin(set(P.id))].groupby("doc_id").size()
    from build_kg import split_articles
    ours = {r.id: len(split_articles(r.id, html_to_lines(r.content_html))) for r in P.itertuples(index=False) if r.content_html}
    common = [d for d in ours if d in kg_n.index]
    res["articles_same_count"] = {"accuracy": round(sum(ours[d] == kg_n[d] for d in common) / max(1, len(common)), 4),
                                  "n": len(common)}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
