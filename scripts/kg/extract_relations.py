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
from build_kg import DOCNUM_RE, html_to_lines  # noqa: E402

PRIORITY = ["replaces", "repeals", "partially_repeals", "amends", "guides", "based_on", "cites"]
SENT_SPLIT = re.compile(r"(?<=[\.;:])\s+")
# numbers after "số", including old formats without a year (15/CP, 37/LCT-HĐNN8, 259-QĐ/NH6)
SO_RE = re.compile(r"\bsố\s+(\d{1,4}[/\-][0-9A-ZĐÐa-zđ/\-\.]*[0-9A-ZĐÐ])")
# documents cited by name and date: "Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015"
NAMED_RE = re.compile(r"\b(Bộ luật|Luật|Pháp lệnh|Hiến pháp|Nghị quyết)\s+([^,;:()]{0,140}?)\s*(?:số[^,;]{0,40})?ngày\s+(\d{1,2})"
                      r"\s*(?:tháng|[/\-])\s*(\d{1,2})\s*(?:năm|[/\-])\s*(\d{4})")
NAMED_TYPES = {"Bộ luật": "Bộ luật", "Luật": "Luật", "Pháp lệnh": "Pháp lệnh", "Hiến pháp": "Hiến pháp", "Nghị quyết": "Nghị quyết"}
TOK = re.compile(r"\w+", re.UNICODE)
DATE_AFTER = re.compile(r"[^;]{0,20}?ngày\s+(\d{1,2})\s*(?:tháng|[/\-])\s*(\d{1,2})\s*(?:năm|[/\-])\s*(\d{4})")


def num_key(num: str) -> str:
    return re.sub(r"[^0-9A-ZĐ]", "", str(num).upper().replace("Ð", "Đ"))


class Resolver:
    """Maps document mentions to document ids: by normalised number, or by type + name + issue date."""

    def __init__(self, docs: pd.DataFrame):
        d = docs.dropna(subset=["id"]).sort_values("ngay_ban_hanh", ascending=False)
        d = d.assign(day=pd.to_datetime(d.ngay_ban_hanh, errors="coerce").dt.date)
        self.by_key = {}  # key -> [(issue date, id)], newest first
        for i, n, day in zip(d.id, d.so_ky_hieu, d.day):
            if isinstance(n, str) and n and n.lower() != "không số":
                self.by_key.setdefault(num_key(n), []).append((day, i))
        self.by_date = {}
        for i, ty, day, title in zip(d.id, d.loai_van_ban, d.day, d.title):
            if ty in NAMED_TYPES.values() and day is not None and not pd.isna(day):
                self.by_date.setdefault((ty, day), []).append((i, set(TOK.findall(str(title).lower()))))
        hp = d[d.loai_van_ban == "Hiến pháp"].dropna(subset=["day"])
        self.constitutions = sorted(zip(hp.day, hp.id))

    def number(self, num: str, day=None, before=None):
        c = self.by_key.get(num_key(num), [])
        if len(c) <= 1:
            return c[0][1] if c else None
        if day is not None:  # date written next to the number
            for d, i in c:
                if d == day:
                    return i
        if before is not None:  # newest document issued before the citing one
            for d, i in c:
                if d is not None and not pd.isna(d) and d <= before:
                    return i
        return c[0][1]

    def named(self, kind: str, name: str, dd: str, mm: str, yyyy: str):
        import datetime
        try:
            day = datetime.date(int(yyyy), int(mm), int(dd))
        except ValueError:
            return None
        cands = self.by_date.get((NAMED_TYPES[kind], day), [])
        nt = set(TOK.findall(f"{kind} {name}".lower()))
        best = max(cands, key=lambda c: len(nt & c[1]) / max(1, len(nt)), default=None)
        if best and (len(cands) == 1 or len(nt & best[1]) / max(1, len(nt)) >= 0.5):
            return best[0]
        return None

    def constitution(self, before):
        ok = [i for d, i in self.constitutions if before is None or d <= before]
        return ok[-1] if ok else None


def mentions(text: str, res: Resolver, issued=None) -> list[str]:
    """Document ids mentioned in a piece of text."""
    import datetime
    out = []
    for m in list(DOCNUM_RE.finditer(text)) + list(SO_RE.finditer(text)):
        day = None
        dm = DATE_AFTER.match(text, m.end())
        if dm:
            try:
                day = datetime.date(int(dm.group(3)), int(dm.group(2)), int(dm.group(1)))
            except ValueError:
                day = None
        i = res.number(m.group(1), day, issued)
        if i:
            out.append(i)
    for m in NAMED_RE.finditer(text):
        i = res.named(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))
        if i:
            out.append(i)
    if re.search(r"\bHiến pháp\b", text) and not NAMED_RE.search(text):
        i = res.constitution(issued)
        if i:
            out.append(i)
    return list(dict.fromkeys(out))


def extract(title: str, html: str, own_id: str | None, res: Resolver, issued=None) -> list[tuple[str, str]]:
    """Returns [(document id, relation type)] for one document."""
    lines = html_to_lines(html) if html else []
    found: dict[str, set] = {}

    def add(ids, ty):
        for i in ids:
            if i != own_id:
                found.setdefault(i, set()).add(ty)

    t = (title or "").lower()
    tm = mentions(title or "", res, issued)
    if "sửa đổi" in t or "bổ sung" in t:
        add(tm, "amends")
    if "quy định chi tiết" in t or "hướng dẫn" in t:
        add(tm, "guides")
    if "bãi bỏ" in t:
        add(tm, "repeals")
    if "thay thế" in t:
        add(tm, "replaces")

    for line in lines:
        low = line.lower()
        if low.startswith("căn cứ"):
            add(mentions(line, res, issued), "based_on")
            continue
        for sent in SENT_SPLIT.split(line):
            sl = sent.lower()
            sn = mentions(sent, res, issued)
            if not sn:
                continue
            if "thay thế" in sl:
                add(sn, "replaces")
            elif "hết hiệu lực" in sl or "bãi bỏ" in sl:
                partial = "bãi bỏ" in sl and re.search(r"(điều|khoản|điểm)\s+\d+", sl.split("bãi bỏ", 1)[-1]) is not None
                add(sn, "partially_repeals" if partial else "repeals")
            elif "sửa đổi" in sl or "bổ sung" in sl:
                add(sn, "amends")
            elif "quy định chi tiết" in sl or "hướng dẫn thi hành" in sl:
                add(sn, "guides")
            else:
                add(sn, "cites")
    out = []
    for i, types in found.items():
        for ty in PRIORITY:  # keep the strongest relation per target document
            if ty in types:
                out.append((i, ty))
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
    docs = pd.read_parquet(Path(args.kg) / "docs.parquet", columns=["id", "so_ky_hieu", "ngay_ban_hanh", "loai_van_ban", "title"])
    extra = pages[~pages.id.isin(docs.id)][["id", "so_ky_hieu", "ngay_ban_hanh", "loai_van_ban", "title"]]
    extra = extra.assign(ngay_ban_hanh=pd.to_datetime(extra.ngay_ban_hanh, errors="coerce"))
    res = Resolver(pd.concat([docs, extra], ignore_index=True))  # crawled documents can also be targets
    rows = []
    for r in pages.itertuples(index=False):
        issued = pd.to_datetime(r.ngay_ban_hanh, errors="coerce")
        for dst, ty in extract(r.title, r.content_html, r.id, res, None if pd.isna(issued) else issued.date()):
            rows.append({"src": r.id, "dst": dst, "type": ty})
    rel = pd.DataFrame(rows, columns=["src", "dst", "type"])
    stats = {"documents": len(pages), "relations": len(rel), "types": rel.type.value_counts().to_dict() if len(rel) else {}}

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
