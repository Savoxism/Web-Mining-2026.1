"""BM25 baseline (bm25s). Tokenization: syllables (default) or pyvi word segmentation.

Usage:
    uv run python scripts/bm25_baseline.py --data data/processed --split test --tokenizer syllable
"""

import argparse
import json
import re
import time
from pathlib import Path

import bm25s
import pandas as pd
from tqdm import tqdm

from metrics import evaluate

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def make_tokenizer(kind: str):
    if kind == "syllable":
        return lambda t: TOKEN_RE.findall(t.lower())
    if kind == "pyvi":
        from pyvi import ViTokenizer

        return lambda t: TOKEN_RE.findall(ViTokenizer.tokenize(t.lower()))
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tokenizer", choices=["syllable", "pyvi"], default="syllable")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--out", default="outputs/baselines")
    args = ap.parse_args()

    data = Path(args.data)
    corpus = pd.read_parquet(data / "corpus.parquet")
    queries = pd.read_parquet(data / f"queries_{args.split}.parquet")
    tok = make_tokenizer(args.tokenizer)

    t0 = time.time()
    corpus_tokens = [tok(t) for t in tqdm(corpus.text.tolist(), desc="tokenize corpus")]
    retriever = bm25s.BM25(k1=args.k1, b=args.b)
    retriever.index(corpus_tokens)
    t_index = time.time() - t0

    q_tokens = [tok(q) for q in tqdm(queries["query"].tolist(), desc="tokenize queries")]
    t1 = time.time()
    idx, _ = retriever.retrieve(q_tokens, k=args.k, show_progress=True, n_threads=-1)
    t_search = time.time() - t1

    cids = corpus.cid.to_numpy()
    ranked = [cids[row].tolist() for row in idx]
    res = evaluate(ranked, queries.cids.tolist())
    res.update(
        {
            "method": f"BM25 ({args.tokenizer})",
            "split": args.split,
            "corpus_size": len(corpus),
            "index_seconds": round(t_index, 1),
            "search_seconds": round(t_search, 1),
        }
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bm25_{args.tokenizer}_{args.split}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
