"""Zero-shot dense retrieval baseline with sentence-transformers (exact search on normalized embeddings).

Usage:
    uv run --extra dense python scripts/dense_baseline.py --model BAAI/bge-m3 --split test
    uv run --extra dense python scripts/dense_baseline.py \
        --model bkai-foundation-models/vietnamese-bi-encoder --segment pyvi --split test
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from metrics import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="pinned model commit SHA")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--segment", choices=["none", "pyvi"], default="none",
                    help="PhoBERT-based models (vietnamese-bi-encoder) expect pyvi word segmentation")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default="outputs/baselines")
    args = ap.parse_args()

    data = Path(args.data)
    corpus = pd.read_parquet(data / "corpus.parquet")
    queries = pd.read_parquet(data / f"queries_{args.split}.parquet")
    docs = corpus.text.tolist()
    qs = queries["query"].tolist()

    if args.segment == "pyvi":
        from pyvi import ViTokenizer

        docs = [ViTokenizer.tokenize(t) for t in docs]
        qs = [ViTokenizer.tokenize(q) for q in qs]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(args.model, device=device, revision=args.revision)
    if device == "cuda":
        model.half()
    model.max_seq_length = min(args.max_seq_len, model.max_seq_length or args.max_seq_len)

    t0 = time.time()
    d_emb = model.encode(docs, batch_size=args.batch_size, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=True)
    q_emb = model.encode(qs, batch_size=args.batch_size, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=True)
    t_enc = time.time() - t0

    cids = corpus.cid.to_numpy()
    ranked = []
    for start in range(0, len(q_emb), 1024):
        scores = q_emb[start : start + 1024] @ d_emb.T
        top = torch.topk(scores, k=args.k, dim=1).indices.cpu().numpy()
        ranked.extend(cids[row].tolist() for row in top)

    res = evaluate(ranked, queries.cids.tolist())
    res.update({
        "method": f"Dense zero-shot ({args.model})",
        "revision": args.revision,
        "split": args.split,
        "segment": args.segment,
        "max_seq_len": model.max_seq_length,
        "corpus_size": len(corpus),
        "device": device,
        "encode_seconds": round(t_enc, 1),
    })
    print(json.dumps(res, ensure_ascii=False, indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.model.split("/")[-1]
    (out / f"dense_{name}_{args.split}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
