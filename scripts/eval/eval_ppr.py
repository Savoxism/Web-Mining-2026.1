"""Personalized PageRank (query-dependent graph signal) on top of the time-aware pipeline.

For each question: bi-encoder top-100 -> top-30 + successors of stale candidates (time-aware, gamma from validation)
-> cross-encoder. PPR is seeded on the documents of these candidates (weights = softmax of cross-encoder scores).
  * re-rank: final = s_CE - gamma * stale + lambda * z(log support(doc))
  * expansion: the best article (bi-encoder) of the top-E PPR documents not yet among the candidates is added and scored
alpha, lambda and E are tuned on validation (MRR@10 against the current gold); test results are reported overall and on
two subsets where graph structure can matter: questions with >= 2 gold articles, and with gold articles in >= 2 documents.

Usage (GPU): .venv/bin/python scripts/eval/eval_ppr.py --bi <bi> --ce <ce> --gamma 2 --out <dir>
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
from lexgraph.pipeline import LexGraph  # noqa: E402
from lexgraph.ppr import DocGraph  # noqa: E402


def mrr_hit(order, gold, k=10):
    for r, x in enumerate(order[:k], start=1):
        if x in gold:
            return 1.0 / r, 1.0
    return 0.0, 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", required=True)
    ap.add_argument("--ce", required=True)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.80)
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--t-eval", default="2026-11-23")
    ap.add_argument("--k-rerank", type=int, default=30)
    ap.add_argument("--max-expand", type=int, default=10)
    ap.add_argument("--val-queries", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.Timestamp(args.t_eval)

    lg = LexGraph(args.bi, args.ce)
    docs = lg.docs
    graph = DocGraph(pd.read_parquet("data/kg/edges.parquet"), list(docs.index))
    rows_by_doc = pd.Series(np.arange(len(lg.corpus))).groupby(lg.doc_ids).apply(list).to_dict()
    print(f"[{time.time()-t0:.0f}s] graph n={graph.n}", flush=True)

    def prepare(queries):
        q_emb = lg.encode_queries(queries["query"].tolist())
        ti, _ = lg.retrieve(q_emb, k=100)
        items = []
        for j, q in enumerate(tqdm(queries.itertuples(index=False), total=len(queries), desc="prepare", mininterval=30)):
            gold = {lg.row_of[c] for c in q.cids if c in lg.row_of}
            rows = lg.candidates(q_emb[j], ti[j], t, args.tau, args.k_rerank, True)
            ce = lg.ce_scores(q.query, rows)
            stale, _, _ = lg.status(rows, t, args.tau)
            g_sorted = sorted(gold)
            g_stale, _, g_succ = lg.status(np.array(g_sorted, dtype=int), t, args.tau)
            cur = set()
            for g, st, s in zip(g_sorted, g_stale, g_succ):
                cur |= set(lg.rows_of_aid[s.succ_aid]) if (st and s is not None and lg.rows_of_aid.get(s.succ_aid)) else {g}
            items.append({"q_vec": q_emb[j], "query": q.query, "rows": np.array(rows), "ce": ce, "stale": stale,
                          "gold": gold, "cur": cur, "multi_gold": len(gold) >= 2,
                          "multi_doc": len({lg.doc_ids[g] for g in gold if lg.doc_ids[g]}) >= 2})
        return items

    def add_ppr(items, alpha):
        for start in range(0, len(items), args.batch):
            chunk = items[start:start + args.batch]
            seeds = []
            for it in chunk:
                p = np.exp(it["ce"] - it["ce"].max()); p /= p.sum()
                sd = {}
                for r, w in zip(it["rows"], p):
                    d = lg.doc_ids[r]
                    if d and d in graph.index:
                        sd[graph.index[d]] = sd.get(graph.index[d], 0.0) + float(w)
                seeds.append(sd)
            R, S = graph.ppr(seeds, alpha=alpha)
            support = R - alpha * S
            for j, it in enumerate(chunk):
                col = support[:, j]
                it[("supp", alpha)] = np.array([col[graph.index[lg.doc_ids[r]]] if lg.doc_ids[r] in graph.index else 0.0
                                                for r in it["rows"]])
                cand_docs = {graph.index[lg.doc_ids[r]] for r in it["rows"] if lg.doc_ids[r] in graph.index}
                top = [i for i in np.argsort(-col)[:200] if i not in cand_docs and col[i] > 0][:args.max_expand]
                it[("exp_docs", alpha)] = [(None, i, float(col[i])) for i in top]

    doc_names = list(docs.index)

    def expand(it, alpha, E):
        """Rows of up to E expansion documents (best article by bi-encoder), cached with CE scores."""
        key = ("exp", alpha, E)
        if key in it:
            return it[key]
        extra = []
        for _, i, sup in it[("exp_docs", alpha)][:E]:
            rows = rows_by_doc.get(doc_names[i], [])
            if not rows:
                continue
            sims = (lg.emb[rows] @ it["q_vec"]).float().cpu().numpy()
            extra.append((rows[int(np.argmax(sims))], sup))
        if extra:
            er = np.array([r for r, _ in extra])
            ce = lg.ce_scores(it["query"], er)
            st, _, _ = lg.status(er, t, args.tau)
            it[key] = (er, ce, st, np.array([s for _, s in extra]))
        else:
            it[key] = (np.array([], dtype=int), np.array([]), np.array([], dtype=bool), np.array([]))
        return it[key]

    def score(items, alpha, lam, E):
        m = {"MRR@10": [], "MRR@10_cur": [], "R@10": [], "MG_MRR@10": [], "MG_R@10": [], "MD_MRR@10": [], "MD_R@10": []}
        for it in items:
            rows, ce, stale = it["rows"], it["ce"], it["stale"]
            supp = it[("supp", alpha)] if lam else np.zeros(len(rows))
            if E:
                er, ece, est, esup = expand(it, alpha, E)
                rows = np.concatenate([rows, er]); ce = np.concatenate([ce, ece])
                stale = np.concatenate([stale, est]); supp = np.concatenate([supp, esup])
            z = np.log(np.maximum(supp, 1e-12))
            z = (z - z.mean()) / (z.std() + 1e-9) if lam else 0.0
            final = ce - args.gamma * stale.astype(float) + lam * z
            order = rows[np.argsort(-final, kind="stable")]
            a, _ = mrr_hit(order, it["gold"]); c, _ = mrr_hit(order, it["cur"])
            r10 = len(set(order[:10]) & it["gold"]) / max(1, len(it["gold"]))
            m["MRR@10"].append(a); m["MRR@10_cur"].append(c); m["R@10"].append(r10)
            if it["multi_gold"]:
                m["MG_MRR@10"].append(a); m["MG_R@10"].append(r10)
            if it["multi_doc"]:
                m["MD_MRR@10"].append(a); m["MD_R@10"].append(r10)
        return {k: round(float(np.mean(v)), 4) if v else None for k, v in m.items()} | {
            "n_multi_gold": len(m["MG_R@10"]), "n_multi_doc": len(m["MD_R@10"])}

    data = Path(args.data)
    val = pd.read_parquet(data / "queries_val.parquet").sample(args.val_queries, random_state=42)
    test = pd.read_parquet(data / "queries_test.parquet")
    v_items, t_items = prepare(val), prepare(test)
    print(f"[{time.time()-t0:.0f}s] prepared", flush=True)
    alphas, lams, Es = [0.15, 0.3, 0.5], [0, 0.1, 0.25, 0.5, 1.0], [0, 3, 10]
    for a in alphas:
        add_ppr(v_items, a); add_ppr(t_items, a)
    print(f"[{time.time()-t0:.0f}s] ppr computed", flush=True)

    grid = []
    for a, lam, E in itertools.product(alphas, lams, Es):
        grid.append({"alpha": a, "lambda": lam, "E": E, **score(v_items, a, lam, E)})
    grid = pd.DataFrame(grid)
    grid.to_csv(out / "val_grid.csv", index=False)
    base_v = grid[(grid["lambda"] == 0) & (grid.E == 0)].iloc[0]
    best = grid.sort_values(["MRR@10_cur", "MRR@10"], ascending=False).iloc[0]
    best_rr = grid[grid.E == 0].sort_values(["MRR@10_cur", "MRR@10"], ascending=False).iloc[0]
    best_ex = grid[grid["lambda"] == 0].sort_values(["MRR@10_cur", "MRR@10"], ascending=False).iloc[0]
    res = {
        "val_base": base_v.to_dict(), "val_best": best.to_dict(),
        "test": {
            "time-aware (no PPR)": score(t_items, 0.3, 0, 0),
            "+ PPR re-rank": score(t_items, best_rr.alpha, best_rr["lambda"], 0) | {"alpha": best_rr.alpha, "lambda": best_rr["lambda"]},
            "+ PPR expansion": score(t_items, best_ex.alpha, 0, int(best_ex.E)) | {"alpha": best_ex.alpha, "E": int(best_ex.E)},
            "+ PPR re-rank + expansion": score(t_items, best.alpha, best["lambda"], int(best.E)) | {
                "alpha": best.alpha, "lambda": best["lambda"], "E": int(best.E)},
        },
        "n_test": len(t_items), "seconds": round(time.time() - t0, 1), "config": vars(args),
    }
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(res, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
