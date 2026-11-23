"""Evaluate the full pipeline (retrieval, re-ranking, graph and time signals) on the extended corpus.

Gold labels
  * original: the passage the question was written for;
  * current:  the same, except that a gold passage superseded at t_eval is replaced by its successor
              article (the answer under the law in force at t_eval).
Temporal subset: questions with at least one gold passage superseded at t_eval.

Usage (GPU):
    .venv/bin/python scripts/eval/eval_pipeline.py --bi BAAI/bge-m3 --bi-revision <sha> \
        --ce BAAI/bge-reranker-v2-m3 --ce-revision <sha> --out outputs/eval/base
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lexgraph.pipeline import LexGraph, Weights  # noqa: E402


def mrr_hit(ranked_rows, gold_rows, k=10):
    for r, x in enumerate(ranked_rows[:k], start=1):
        if x in gold_rows:
            return 1.0 / r, 1.0
    return 0.0, 0.0


def recall(ranked_rows, gold_rows, k):
    return len(set(ranked_rows[:k]) & gold_rows) / max(1, len(gold_rows))


def prepare(lg: LexGraph, queries: pd.DataFrame, t, tau, k_retrieve, k_rerank, use_ce):
    """Candidates, cross-encoder scores and features for each query (computed once)."""
    q_emb = lg.encode_queries(queries["query"].tolist())
    ti, ts = lg.retrieve(q_emb, k=k_retrieve)
    items = []
    for j, q in enumerate(tqdm(queries.itertuples(index=False), total=len(queries), desc="rerank", mininterval=30)):
        gold = {lg.row_of[c] for c in q.cids if c in lg.row_of}
        rows = lg.candidates(q_emb[j], ti[j], t, tau, k_rerank, inject=True)
        bi = (lg.emb[rows] @ q_emb[j]).float().cpu().numpy()
        ce = lg.ce_scores(q.query, rows) if use_ce else None
        stale, _, _ = lg.status(rows, t, tau)
        # current gold: superseded gold passages -> rows of their successor article
        g_stale, _, g_succ = lg.status(np.array(sorted(gold)), t, tau) if gold else (np.array([]), [], [])
        cur = set()
        temporal = False
        for g, st, s in zip(sorted(gold), g_stale, g_succ):
            if st and s is not None and lg.rows_of_aid.get(s.succ_aid):
                cur |= set(lg.rows_of_aid[s.succ_aid]); temporal = True
            else:
                cur.add(g)
        items.append({"rows": rows, "bi": bi, "ce": ce, "stale": stale,
                      "prop_ce": lg.propagation(rows, ce) if use_ce else None,
                      "prop_bi": lg.propagation(rows, bi * 10),
                      "retrieved": ti[j], "gold": gold, "cur": cur, "temporal": temporal,
                      "injected": len(rows) > k_rerank})
    return items


def score(lg, items, w: Weights, use_ce: bool, k_rerank: int):
    m = {"MRR@10": [], "R@10": [], "MRR@10_cur": [], "Hit@10_cur": [], "Stale@10": [],
         "T_MRR@10_cur": [], "T_Hit@10_cur": [], "T_Stale@10": []}
    for it in items:
        if use_ce:
            base, prop = it["ce"], it["prop_ce"]
            rows, stale = it["rows"], it["stale"]
        else:
            base, prop = it["bi"] * 10, it["prop_bi"]
            rows, stale = it["rows"], it["stale"]
        if not w.inject:
            keep = slice(0, min(k_rerank, len(rows)))
            rows, base, prop, stale = rows[keep], base[keep], prop[keep], stale[keep]
        final = lg.combine(rows, base, stale, prop, w)
        order = rows[np.argsort(-final, kind="stable")]
        top_stale = stale[np.argsort(-final, kind="stable")][:10]
        a, _ = mrr_hit(order, it["gold"])
        m["MRR@10"].append(a)
        m["R@10"].append(recall(order, it["gold"], 10))
        c, h = mrr_hit(order, it["cur"])
        m["MRR@10_cur"].append(c); m["Hit@10_cur"].append(h)
        m["Stale@10"].append(float(top_stale.mean()) if len(top_stale) else 0.0)
        if it["temporal"]:
            m["T_MRR@10_cur"].append(c); m["T_Hit@10_cur"].append(h)
            m["T_Stale@10"].append(float(top_stale.mean()))
    return {k: round(float(np.mean(v)), 4) if v else None for k, v in m.items()}


def retrieval_only(lg, items, k=100):
    m = {"MRR@10": [], "R@10": [], "R@100": []}
    for it in items:
        order = it["retrieved"]
        a, _ = mrr_hit(order, it["gold"])
        m["MRR@10"].append(a)
        m["R@10"].append(recall(order, it["gold"], 10))
        m["R@100"].append(recall(order, it["gold"], k))
    return {k: round(float(np.mean(v)), 4) for k, v in m.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", required=True)
    ap.add_argument("--bi-revision", default=None)
    ap.add_argument("--ce", default=None)
    ap.add_argument("--ce-revision", default=None)
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--t-eval", default="2026-11-23")
    ap.add_argument("--k-retrieve", type=int, default=100)
    ap.add_argument("--k-rerank", type=int, default=30)
    ap.add_argument("--tau", type=float, default=0.80)
    ap.add_argument("--val-queries", type=int, default=1500)
    ap.add_argument("--test-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.Timestamp(args.t_eval)
    use_ce = args.ce is not None

    lg = LexGraph(args.bi, args.ce, bi_revision=args.bi_revision, ce_revision=args.ce_revision)
    data = Path(args.data)
    val = pd.read_parquet(data / "queries_val.parquet").sample(args.val_queries, random_state=42)
    test = pd.read_parquet(data / "queries_test.parquet")
    if args.test_queries:
        test = test.sample(args.test_queries, random_state=42)

    print(f"[{time.time()-t0:.0f}s] preparing val ({len(val)}) and test ({len(test)})", flush=True)
    v_items = prepare(lg, val, t, args.tau, args.k_retrieve, args.k_rerank, use_ce)
    t_items = prepare(lg, test, t, args.tau, args.k_retrieve, args.k_rerank, use_ce)

    # grid search on val, objective = MRR@10 against the current gold
    grid = []
    for beta, gamma, mu in itertools.product([0, 0.1, 0.25, 0.5, 1.0], [0, 0.5, 1, 2, 4, 8], [0, 0.5, 1, 2]):
        w = Weights(beta=beta, gamma=gamma, mu=mu, tau=args.tau, inject=True)
        r = score(lg, v_items, w, use_ce, args.k_rerank)
        grid.append({"beta": beta, "gamma": gamma, "mu": mu, **r})
    grid = pd.DataFrame(grid)
    grid.to_csv(out / "val_grid.csv", index=False)
    best = grid.sort_values(["MRR@10_cur", "MRR@10"], ascending=False).iloc[0]
    bg = grid[grid.gamma == 0].sort_values("MRR@10_cur", ascending=False).iloc[0]       # graph only
    tg = grid[(grid.beta == 0) & (grid.mu == 0)].sort_values("MRR@10_cur", ascending=False).iloc[0]  # time only

    variants = {
        "retriever only": Weights(use_ce=False, inject=False),
        ("+ cross-encoder" if use_ce else "retriever (rank by bi score)"): Weights(inject=False),
        "+ graph (PageRank, propagation)": Weights(beta=bg.beta, mu=bg.mu, inject=False),
        "+ time (stale penalty, successor injection)": Weights(gamma=tg.gamma, inject=True),
        "full (graph + time)": Weights(beta=best.beta, gamma=best.gamma, mu=best.mu, inject=True),
    }
    results = {"retrieval_only_test": retrieval_only(lg, t_items), "variants": {}}
    for name, w in variants.items():
        ce_flag = use_ce and w.use_ce
        results["variants"][name] = {"weights": {"beta": w.beta, "gamma": w.gamma, "mu": w.mu, "inject": w.inject},
                                     **score(lg, t_items, w, ce_flag, args.k_rerank)}
    results["n_test"] = len(t_items)
    results["n_test_temporal"] = int(sum(it["temporal"] for it in t_items))
    results["n_val"] = len(v_items)
    results["best_val"] = best.to_dict()
    results["config"] = vars(args)
    results["seconds"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(results, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
