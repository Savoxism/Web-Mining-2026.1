"""Mine semi-hard negatives with a bi-encoder.

For each (query, positive) pair, candidates come from the bi-encoder top-k. A candidate is a
semi-hard negative when it is not a gold passage, ranks at or below --min-rank, and scores below
sim(q, positive) - margin (very close candidates are likely unlabelled positives).

Usage (GPU):
    .venv/bin/python scripts/train/mine_negatives.py --model BAAI/bge-m3 --revision <sha> \
        --split train --n-neg 3 --out data/train/biencoder_train.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--min-rank", type=int, default=3, help="skip the top (min_rank-1) non-gold candidates")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--n-neg", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-candidates", default=None, help="also save the top-k candidate cids per query")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = Path(args.data)
    corpus = pd.read_parquet(data / "corpus.parquet")
    queries = pd.read_parquet(data / f"queries_{args.split}.parquet")
    if args.max_queries and len(queries) > args.max_queries:
        queries = queries.sample(args.max_queries, random_state=args.seed).reset_index(drop=True)
    cid2row = {c: i for i, c in enumerate(corpus.cid.values)}
    cids = corpus.cid.to_numpy()
    texts = corpus.text.to_numpy()

    model = SentenceTransformer(args.model, revision=args.revision, device="cuda")
    model.half()
    model.max_seq_length = args.max_seq_len
    d_emb = model.encode(corpus.text.tolist(), batch_size=args.batch_size, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=True)
    q_emb = model.encode(queries["query"].tolist(), batch_size=args.batch_size, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=True)

    rows, stats = [], {"pairs": 0, "pairs_with_full_negatives": 0, "neg_rank_mean": []}
    dump = []
    for start in range(0, len(q_emb), 2048):
        s = q_emb[start:start + 2048] @ d_emb.T
        top_s, top_i = torch.topk(s, k=args.k, dim=1)
        top_s, top_i = top_s.float().cpu().numpy(), top_i.cpu().numpy()
        for j in range(len(top_i)):
            q = queries.iloc[start + j]
            gold = set(int(c) for c in q.cids)
            gold_rows = [cid2row[c] for c in gold if c in cid2row]
            if args.dump_candidates:
                dump.append({"qid": int(q.qid), "query": q["query"], "cids": sorted(gold),
                             "cand_cids": [int(cids[r]) for r in top_i[j]]})
            cand = [(r, float(sc)) for r, sc in zip(top_i[j], top_s[j]) if int(cids[r]) not in gold]
            for p_row in gold_rows:
                p_score = float(q_emb[start + j] @ d_emb[p_row])
                pool = [(rank, r) for rank, (r, sc) in enumerate(cand, start=1)
                        if rank >= args.min_rank and sc < p_score - args.margin]
                if len(pool) < args.n_neg:  # fall back to the hardest remaining non-gold candidates
                    pool = [(rank, r) for rank, (r, sc) in enumerate(cand, start=1) if rank >= args.min_rank]
                pick = sorted(rng.choice(len(pool), size=min(args.n_neg, len(pool)), replace=False)) if pool else []
                negs = [pool[k][1] for k in pick]
                stats["pairs"] += 1
                stats["pairs_with_full_negatives"] += len(negs) == args.n_neg
                stats["neg_rank_mean"].extend(pool[k][0] for k in pick)
                if len(negs) < args.n_neg:
                    continue
                row = {"qid": int(q.qid), "anchor": q["query"], "positive": texts[p_row],
                       "pos_cid": int(cids[p_row])}
                for k, r in enumerate(negs, start=1):
                    row[f"negative_{k}"] = texts[r]
                    row[f"neg_cid_{k}"] = int(cids[r])
                rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.dump_candidates:
        pd.DataFrame(dump).to_parquet(args.dump_candidates, index=False)
    pd.DataFrame(rows).to_parquet(out, index=False)
    stats["neg_rank_mean"] = round(float(np.mean(stats["neg_rank_mean"])), 2) if stats["neg_rank_mean"] else None
    stats["rows"] = len(rows)
    stats.update(vars(args))
    out.with_suffix(".json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
