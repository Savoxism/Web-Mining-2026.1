"""Polite incremental crawler for vbpl.vn document pages (robots-allowed paths only).

* URLs come from the public sitemap (https://vbpl.vn/sitemap.xml), with `lastmod` for incremental runs.
* Only /van-ban/chi-tiet/<slug>--<id> pages are fetched; robots.txt (Disallow: /api/, /Pages/) is checked for every URL.
* One request per `--delay` seconds, generic research User-Agent, on-disk cache (gzip), retries with backoff.
* Each page is server-rendered with a Next.js payload (`self.__next_f`) that contains the document metadata
  (`initialDoc`) and the full text HTML; relation lists are NOT in the page (they come from the disallowed API), so
  relations are extracted from the text afterwards (scripts/kg/extract_relations.py).

Usage:
    .venv/bin/python scripts/crawl/crawl_vbpl.py --sitemap data/crawl/sitemap_central.parquet --select new \
        --out data/crawl/pages_new.parquet
    .venv/bin/python scripts/crawl/crawl_vbpl.py --select known --n 300 --out data/crawl/pages_known.parquet
"""

import argparse
import gzip
import json
import re
import time
import urllib.request
import urllib.robotparser
from pathlib import Path

import pandas as pd
from tqdm import tqdm

UA = "Mozilla/5.0 (compatible; research-crawler; university course project)"
PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', re.S)


def payload_text(html: str) -> str:
    out = []
    for c in PUSH_RE.findall(html):
        try:
            out.append(json.loads('"' + c + '"'))
        except json.JSONDecodeError:
            continue
    return "".join(out)


def parse_page(html: str) -> dict | None:
    txt = payload_text(html)
    i = txt.find('"initialDoc":')
    if i < 0:
        return None
    obj, _ = json.JSONDecoder().raw_decode(txt[i + len('"initialDoc":'):])
    content = ""
    ref = ((obj.get("documentContent") or {}).get("content") or "")
    if isinstance(ref, str) and ref.startswith("$"):
        key = ref[1:]
        m = re.search(r"(?:^|\n)" + re.escape(key) + r":T[0-9a-f]+,", txt)
        if m:
            start = m.end()
            end = txt.find("</html>", start)
            content = txt[start:end + 7] if end > 0 else txt[start:start + 2_000_000]
    if not content:  # fall back to the largest <html> block of the payload
        blocks = re.findall(r"<html.*?</html>", txt, re.S)
        content = max(blocks, key=len) if blocks else ""
    g = lambda d, k: (d or {}).get(k)  # noqa: E731
    return {
        "id": str(obj.get("id")), "so_ky_hieu": obj.get("docNum"), "title": obj.get("title"),
        "loai_van_ban": g(obj.get("docType"), "name"), "ngay_ban_hanh": obj.get("issueDate"),
        "ngay_co_hieu_luc": obj.get("effFrom"), "ngay_het_hieu_luc": obj.get("effTo"),
        "tinh_trang_hieu_luc": g(obj.get("effStatus"), "name"), "co_quan_ban_hanh": obj.get("agencyName"),
        "linh_vuc": "; ".join(f.get("name", "") for f in (obj.get("documentFields") or [])),
        "updated": obj.get("updatedDate"), "is_translation": obj.get("isTranslationDoc"),
        "is_consolidated": obj.get("isConsolidatedDocument"), "content_html": content,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sitemap", default="data/crawl/sitemap_central.parquet")
    ap.add_argument("--known-ids", default="data/vbpl/data/metadata.parquet")
    ap.add_argument("--select", choices=["new", "known"], default="new")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--cache", default="data/crawl/cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # fetch robots.txt with our User-Agent (the default Python UA gets 403, which robotparser treats as "disallow all")
    rp = urllib.robotparser.RobotFileParser()
    req = urllib.request.Request("https://vbpl.vn/robots.txt", headers={"User-Agent": UA})
    rp.parse(urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore").splitlines())
    assert not rp.can_fetch(UA, "https://vbpl.vn/api/x") and rp.can_fetch(UA, "https://vbpl.vn/van-ban/chi-tiet/x--1")
    sm = pd.read_parquet(args.sitemap)
    known = set(pd.read_parquet(args.known_ids, columns=["id"]).id)
    sel = sm[~sm.doc_id.isin(known)] if args.select == "new" else sm[sm.doc_id.isin(known)]
    sel = sel.drop_duplicates("doc_id")
    if args.n and len(sel) > args.n:
        sel = sel.sample(args.n, random_state=args.seed)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    rows, stats = [], {"fetched": 0, "cached": 0, "robots_blocked": 0, "failed": 0, "parsed": 0}
    for r in tqdm(sel.itertuples(index=False), total=len(sel), desc="crawl", mininterval=30):
        if "/van-ban/chi-tiet/" not in r.url or not rp.can_fetch(UA, r.url):
            stats["robots_blocked"] += 1
            continue
        cp = cache / f"{r.doc_id}.html.gz"
        if cp.exists():
            html = gzip.decompress(cp.read_bytes()).decode("utf-8", "ignore")
            stats["cached"] += 1
        else:
            html = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(r.url, headers={"User-Agent": UA, "Accept-Language": "vi"})
                    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
                    break
                except Exception:
                    time.sleep(args.delay * (2 ** (attempt + 1)))
            time.sleep(args.delay)
            if html is None:
                stats["failed"] += 1
                continue
            cp.write_bytes(gzip.compress(html.encode("utf-8")))
            stats["fetched"] += 1
        try:
            d = parse_page(html)
        except Exception:
            d = None
        if d:
            d.update({"url": r.url, "lastmod": r.lastmod})
            rows.append(d)
            stats["parsed"] += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    stats["with_content"] = int(sum(bool(x["content_html"]) for x in rows))
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
