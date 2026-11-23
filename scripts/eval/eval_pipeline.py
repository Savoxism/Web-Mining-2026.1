"""Evaluate the full pipeline (retrieval, re-ranking, graph and time signals) on the extended corpus.

Gold labels
  * original: the passage the question was written for;
  * current:  the same, except that a gold passage superseded at t_eval is replaced by its successor
              article (the answer under the law in force at t_eval), using the main configuration.
Temporal subset: questions with at least one gold passage superseded at t_eval.

Ablation configurations change how successors are accepted (explicit references, edge types); each
configuration has its own injected candidates and stale flags, while the current gold stays fixed.

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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lexgraph.pipeline import LexGraph, Weights  # noqa: E402

CONFIGS = {
    "main": {"use_explicit": True, "allowed_types": None},
    "no_explicit": {"use_explicit": False, "allowed_types": None},
    "amends_only": {"use_explicit": True, "allowed_types": {"amends"}},
    "replaces_repeals_only": {"use_explicit": True, "allowed_types": {"replaces", "repeals", "partially_repeals"}},
}


def mrr_hit(order, gold, k=10):
    for r, x in enumerate(order[:k], start=1):
        if x in gold:
            return 1.0 / r, 1.0
    return 0.0, 0.0


def recall(order, gold, k):
    return len(set(order[:k]) & gold) / max(1, len(gold))


def prepare(lg: LexGraph, queries: pd.DataFrame, t, tau, k_retrieve, k_rerank, use_ce, configs):
    q_emb = lg.encode_queries(queries["query"].tolist())
    ti, _ = lg.retrieve(q_emb, k=k_retrieve)
    items = []
    for j, q in enumerate(tqdm(queries.itertuples(index=False), total=len(queries), desc="prepare", mininterval=30)):
        gold = {lg.row_of[c] for c in q.cids if c in lg.row_of}
        cand = {name: lg.candidates(q_emb[j], ti[j], t, tau, k_rerank, True, **cfg) for name, cfg in configs.items()}
        union = list(dict.fromkeys(int(r) for rows in cand.values() for r in rows))
        rows = np.array(union, dtype=int)
        pos = {r: k for k, r in enumerate(union)}
        bi = (lg.emb[rows] @ q_emb[j]).float().cpu().numpy()
        ce = lg.ce_scores(q.query, rows) if use_ce else bi * 10
        cfg_data = {}
        for name, cfg in configs.items():
            idx = np.array([pos[int(r)] for r in cand[name]], dtype=int)
            stale, _, _ = lg.status(rows[idx], t, tau, **cfg)
            cfg_data[name] = {"idx": idx, "stale": stale}
        # current gold (main configuration)
        g_sorted = sorted(gold)
        g_stale, _, g_succ = lg.status(np.array(g_sorted, dtype=int), t, tau)
        cur, temporal = set(), False
        for g, st, s in zip(g_sorted, g_stale, g_succ):
            if st and s is not None and lg.rows_of_aid.get(s.succ_aid):
                cur |= set(lg.rows_of_aid[s.succ_aid]); temporal = True
            else:
                cur.add(g)
        items.append({"rows": rows, "bi": bi, "ce": ce, "cfg": cfg_data, "retrieved": ti[j], "gold": gold,
                      "cur": cur, "temporal": temporal, "gold_stale": bool(np.any(g_stale)) if len(g_stale) else False})
    return items


def score(lg, items, w: Weights, k_rerank: int, cfg: str = "main", use_ce: bool = True):
    m = {k: [] for k in ["MRR@10", "R@10", "MRR@10_cur", "Hit@10_cur", "Stale@10",
                         "T_MRR@10_cur", "T_Hit@10_cur", "T_Stale@10"]}
    for it in items:
        d = it["cfg"][cfg]
        idx = d["idx"] if w.inject else d["idx"][:k_rerank]
        stale = d["stale"] if w.inject else d["stale"][:k_rerank]
        rows = it["rows"][idx]
        base = (it["ce"] if use_ce else it["bi"] * 10)[idx]
        key = ("prop", cfg, w.inject, use_ce)
        if w.mu and key not in it:
            it[key] = lg.propagation(rows, base)
        prop = it[key] if w.mu else np.zeros(len(rows))
        final = lg.combine(rows, base, stale, prop, w)
        o = np.argsort(-final, kind="stable")
        order, top_stale = rows[o], stale[o][:10]
        a, _ = mrr_hit(order, it["gold"])
        c, h = mrr_hit(order, it["cur"])
        m["MRR@10"].append(a); m["R@10"].append(recall(order, it["gold"], 10))
        m["MRR@10_cur"].append(c); m["Hit@10_cur"].append(h)
        m["Stale@10"].append(float(top_stale.mean()) if len(top_stale) else 0.0)
        if it["temporal"]:
            m["T_MRR@10_cur"].append(c); m["T_Hit@10_cur"].append(h)
            m["T_Stale@10"].append(float(top_stale.mean()) if len(top_stale) else 0.0)
    return {k: round(float(np.mean(v)), 4) if v else None for k, v in m.items()}


def retrieval_only(items, k=100):
    m = {"MRR@10": [], "R@10": [], "R@100": [], "MRR@10_cur": [], "T_MRR@10_cur": []}
    for it in items:
        order = list(it["retrieved"])
        a, _ = mrr_hit(order, it["gold"]); c, _ = mrr_hit(order, it["cur"])
        m["MRR@10"].append(a); m["R@10"].append(recall(order, it["gold"], 10)); m["R@100"].append(recall(order, it["gold"], k))
        m["MRR@10_cur"].append(c)
        if it["temporal"]:
            m["T_MRR@10_cur"].append(c)
    return {k: round(float(np.mean(v)), 4) if v else None for k, v in m.items()}


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
    ap.add_argument("--no-graph", action="store_true", help="time signals only (beta = mu = 0 everywhere)")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.Timestamp(args.t_eval)
    use_ce = args.ce is not None

    lg = LexGraph(args.bi, args.ce, bi_revision=args.bi_revision, ce_revision=args.ce_revision)
    print(f"[{time.time()-t0:.0f}s] loaded, corpus {len(lg.corpus)}", flush=True)
    data = Path(args.data)
    val = pd.read_parquet(data / "queries_val.parquet").sample(args.val_queries, random_state=42)
    test = pd.read_parquet(data / "queries_test.parquet")
    if args.test_queries:
        test = test.sample(args.test_queries, random_state=42)

    v_items = prepare(lg, val, t, args.tau, args.k_retrieve, args.k_rerank, use_ce, {"main": CONFIGS["main"]})
    print(f"[{time.time()-t0:.0f}s] val prepared", flush=True)
    t_items = prepare(lg, test, t, args.tau, args.k_retrieve, args.k_rerank, use_ce, CONFIGS)
    print(f"[{time.time()-t0:.0f}s] test prepared", flush=True)

    grid = []
    betas, mus = ([0], [0]) if args.no_graph else ([0, 0.1, 0.25, 0.5, 1.0], [0, 0.5, 1, 2])
    for beta, gamma, mu in itertools.product(betas, [0, 0.5, 1, 2, 4, 8], mus):
        r = score(lg, v_items, Weights(beta=beta, gamma=gamma, mu=mu, tau=args.tau, inject=True), args.k_rerank, use_ce=use_ce)
        grid.append({"beta": beta, "gamma": gamma, "mu": mu, **r})
    grid = pd.DataFrame(grid)
    grid.to_csv(out / "val_grid.csv", index=False)
    best = grid.sort_values(["MRR@10_cur", "MRR@10"], ascending=False).iloc[0]
    bg = grid[grid.gamma == 0].sort_values(["MRR@10_cur", "MRR@10"], ascending=False).iloc[0]
    tg = grid[(grid.beta == 0) & (grid.mu == 0)].sort_values(["MRR@10_cur", "MRR@10"], ascending=False).iloc[0]
    B, G, M = float(best.beta), float(best.gamma), float(best.mu)

    V = []
    V.append(("retriever + re-ranker", Weights(inject=False), "main"))
    if not args.no_graph:
        V.append(("+ graph (beta, mu)", Weights(beta=float(bg.beta), mu=float(bg.mu), inject=False), "main"))
    V.append(("+ time (gamma, injection)", Weights(gamma=float(tg.gamma), inject=True), "main"))
    V.append(("full (graph + time)", Weights(beta=B, gamma=G, mu=M, inject=True), "main"))
    if not args.no_graph:
        V.append(("full - PageRank", Weights(beta=0, gamma=G, mu=M, inject=True), "main"))
        V.append(("full - propagation", Weights(beta=B, gamma=G, mu=0, inject=True), "main"))
    V.append(("full - successor injection", Weights(beta=B, gamma=G, mu=M, inject=False), "main"))
    V.append(("full - explicit references", Weights(beta=B, gamma=G, mu=M, inject=True), "no_explicit"))
    V.append(("full, amends edges only", Weights(beta=B, gamma=G, mu=M, inject=True), "amends_only"))
    V.append(("full, replaces/repeals only", Weights(beta=B, gamma=G, mu=M, inject=True), "replaces_repeals_only"))

    results = {"retrieval_only_test": retrieval_only(t_items), "variants": {}}
    for name, w, cfg in V:
        results["variants"][name] = {"weights": {"beta": w.beta, "gamma": w.gamma, "mu": w.mu, "inject": w.inject},
                                     "config": cfg, **score(lg, t_items, w, args.k_rerank, cfg, use_ce)}
    results["n_test"] = len(t_items)
    results["n_test_temporal"] = int(sum(it["temporal"] for it in t_items))
    results["n_test_gold_stale"] = int(sum(it["gold_stale"] for it in t_items))
    results["n_val"] = len(v_items)
    results["best_val"] = {k: (float(v) if v is not None else None) for k, v in best.to_dict().items()}
    results["config"] = vars(args)
    results["seconds"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(results, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
