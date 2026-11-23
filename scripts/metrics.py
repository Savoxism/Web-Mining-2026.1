"""Ranking metrics for retrieval evaluation."""

import math


def evaluate(ranked: list[list[int]], relevant: list[list[int]], ks=(1, 10, 50, 100)) -> dict:
    """ranked[i]: retrieved cids for query i (best first); relevant[i]: gold cids."""
    n = len(ranked)
    out = {"n_queries": n}
    mrr10 = 0.0
    ndcg10 = 0.0
    recall = {k: 0.0 for k in ks}
    for r, g in zip(ranked, relevant):
        gold = set(g)
        for rank, c in enumerate(r[:10], start=1):
            if c in gold:
                mrr10 += 1.0 / rank
                break
        dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(r[:10]) if c in gold)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
        ndcg10 += dcg / idcg if idcg else 0.0
        for k in ks:
            recall[k] += len(gold & set(r[:k])) / len(gold)
    out["MRR@10"] = round(mrr10 / n, 4)
    out["NDCG@10"] = round(ndcg10 / n, 4)
    for k in ks:
        out[f"Recall@{k}"] = round(recall[k] / n, 4)
    return out
