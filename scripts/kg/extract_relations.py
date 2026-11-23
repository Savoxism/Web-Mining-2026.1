"""Rule-based relation extraction from the text of crawled legal documents.

The vbpl.vn document pages do not contain relation lists (those come from the API, disallowed by robots.txt), so we
read them from the text, as a reader would:
  * replaces:          "... thay thế Nghị định số X ..."
  * repeals:           "Bãi bỏ Thông tư số X", "X hết hiệu lực (thi hành) kể từ ..."
  * partially_repeals: "Bãi bỏ Điều 5, khoản 2 Điều 7 của Thông tư số X"
  * amends:            "Sửa đổi, bổ sung (một số điều của) Nghị định số X" (title or body)
  * guides:            "Quy định chi tiết / hướng dẫn (thi hành) ... Luật số X" (title)
  * based_on:          lines starting with "Căn cứ"
  * cites:             any other document number in the body
Evaluated against the portal's own relations on documents present in both our crawl and the public crawl.

Usage:
    .venv/bin/python scripts/kg/extract_relations.py --pages data/crawl/pages_known.parquet --evaluate
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kg import DOCNUM_RE, RELATION_MAP, html_to_lines  # noqa: E402

PRIORITY = ["replaces", "repeals", "partially_repeals", "amends", "guides", "based_on", "cites"]
SENT_SPLIT = re.compile(r"(?<=[\.;:])\s+")


def extract(title: str, html: str, own_num: str | None) -> list[tuple[str, str]]:
    """Returns [(document number, relation type)] for one document."""
    lines = html_to_lines(html) if html else []
    found: dict[str, set] = {}

    def add(num, ty):
        if own_num and num == own_num:
            return
        found.setdefault(num, set()).add(ty)

    t = (title or "").lower()
    title_nums = DOCNUM_RE.findall(title or "")
    if "sửa đổi" in t or "bổ sung" in t:
        for n in title_nums:
            add(n, "amends")
    if "quy định chi tiết" in t or "hướng dẫn" in t:
        for n in title_nums:
            add(n, "guides")
    if "bãi bỏ" in t:
        for n in title_nums:
            add(n, "repeals")
    if "thay thế" in t:
        for n in title_nums:
            add(n, "replaces")

    for line in lines:
        low = line.lower()
        nums = DOCNUM_RE.findall(line)
        if not nums:
            continue
        if low.startswith("căn cứ"):
            for n in nums:
                add(n, "based_on")
            continue
        for sent in SENT_SPLIT.split(line):
            sl = sent.lower()
            sn = DOCNUM_RE.findall(sent)
            if not sn:
                continue
            if "thay thế" in sl:
                for n in sn:
                    add(n, "replaces")
            elif "hết hiệu lực" in sl or "bãi bỏ" in sl:
                partial = re.search(r"(điều|khoản|điểm)\s+\d+", sl.split("bãi bỏ", 1)[-1]) is not None and "bãi bỏ" in sl
                for n in sn:
                    add(n, "partially_repeals" if partial else "repeals")
            elif "sửa đổi" in sl or "bổ sung" in sl:
                for n in sn:
                    add(n, "amends")
            elif "quy định chi tiết" in sl or "hướng dẫn thi hành" in sl:
                for n in sn:
                    add(n, "guides")
            else:
                for n in sn:
                    add(n, "cites")
    out = []
    for n, types in found.items():
        for ty in PRIORITY:  # keep the strongest relation per target document
            if ty in types:
                out.append((n, ty))
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True)
    ap.add_argument("--kg", default="data/kg")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pages = pd.read_parquet(args.pages)
    docs = pd.read_parquet(Path(args.kg) / "docs.parquet", columns=["id", "so_ky_hieu", "ngay_ban_hanh"])
    num2doc = (docs.sort_values("ngay_ban_hanh", ascending=False).dropna(subset=["so_ky_hieu"])
               .drop_duplicates("so_ky_hieu").set_index("so_ky_hieu").id.to_dict())
    # crawled documents can also be targets (newest first)
    for r in pages.itertuples(index=False):
        if r.so_ky_hieu and r.so_ky_hieu not in num2doc:
            num2doc[r.so_ky_hieu] = r.id
    rows = []
    for r in pages.itertuples(index=False):
        for num, ty in extract(r.title, r.content_html, r.so_ky_hieu):
            dst = num2doc.get(num)
            rows.append({"src": r.id, "dst_num": num, "dst": dst, "type": ty})
    rel = pd.DataFrame(rows)
    stats = {"documents": len(pages), "relations": len(rel), "resolved": round(float(rel.dst.notna().mean()), 4) if len(rel) else 0,
             "types": rel.type.value_counts().to_dict() if len(rel) else {}}

    if args.evaluate:
        E = pd.read_parquet(Path(args.kg) / "edges.parquet")
        src = set(pages.id)
        gold = E[E.src.isin(src) & E.type.isin(PRIORITY)][["src", "dst", "type"]].drop_duplicates()
        pred = rel.dropna(subset=["dst"])[["src", "dst", "type"]].drop_duplicates()
        ev = {}
        gp, pp = set(map(tuple, gold[["src", "dst"]].values)), set(map(tuple, pred[["src", "dst"]].values))
        tp = len(gp & pp)
        ev["any_type"] = {"precision": round(tp / max(1, len(pp)), 4), "recall": round(tp / max(1, len(gp)), 4),
                          "gold": len(gp), "pred": len(pp)}
        for ty in PRIORITY:
            g = set(map(tuple, gold[gold.type == ty][["src", "dst"]].values))
            p = set(map(tuple, pred[pred.type == ty][["src", "dst"]].values))
            if g or p:
                ev[ty] = {"precision": round(len(g & p) / max(1, len(p)), 4), "recall": round(len(g & p) / max(1, len(g)), 4),
                          "gold": len(g), "pred": len(p)}
        # typed edges that matter for validity: succession (replaces/repeals/partially_repeals/amends)
        succ = ["replaces", "repeals", "partially_repeals", "amends"]
        g = set(map(tuple, gold[gold.type.isin(succ)][["src", "dst"]].values))
        p = set(map(tuple, pred[pred.type.isin(succ)][["src", "dst"]].values))
        ev["succession_edges"] = {"precision": round(len(g & p) / max(1, len(p)), 4), "recall": round(len(g & p) / max(1, len(g)), 4),
                                  "gold": len(g), "pred": len(p)}
        stats["evaluation_vs_portal"] = ev
    if args.out:
        rel.to_parquet(args.out, index=False)
        Path(args.out).with_suffix(".json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
