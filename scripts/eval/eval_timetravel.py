"""Time-travel benchmark: the same question asked before and after the law changed.

For every test question whose gold article g was superseded by an article s (accepted alignment, main
configuration), we ask it at two dates:
  * t_before = the day before the successor document took effect (if g's document was already in force):
    the correct answer is g;
  * t_after  = t_eval (2026-11-23): the correct answer is s.
Systems: a static RAG over the original QA corpus (no validity, same answer at every date), the extended
corpus without time signals, and LexGraph-VN (time-aware, best validation weights).
Metrics per date: Hit@1, Hit@10, MRR@10 against the date-specific gold, Stale@10 at that date, and the share of
questions answered correctly (Hit@10) at both dates.

Usage:
    .venv/bin/python scripts/eval/eval_timetravel.py --bi <bi> --ce <ce> --weights <eval results.json> --out <dir>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lexgraph.pipeline import LexGraph, Weights  # noqa: E402
from lexgraph.temporal import in_force  # noqa: E402


def ranks(order, gold, k=10):
    for r, x in enumerate(order[:k], start=1):
        if x in gold:
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", required=True)
    ap.add_argument("--ce", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--t-eval", default="2026-11-23")
    ap.add_argument("--k-retrieve", type=int, default=100)
    ap.add_argument("--k-rerank", type=int, default=30)
    ap.add_argument("--max-questions", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_after = pd.Timestamp(args.t_eval)
    best = json.loads(Path(args.weights).read_text())["best_val"]
    w = Weights(beta=best["beta"], gamma=best["gamma"], mu=best["mu"], inject=True)
    lg = LexGraph(args.bi, args.ce)
    is_qa = lg.corpus.source.to_numpy() == "qa"

    # ---------------- build the benchmark
    test = pd.read_parquet(Path(args.data) / "queries_test.parquet")
    items = []
    for q in test.itertuples(index=False):
        gold = sorted(lg.row_of[c] for c in q.cids if c in lg.row_of)
        for g in gold:
            s = lg.best_successor(g, t_after, w.tau)
            if s is None or not lg.rows_of_aid.get(s.succ_aid) or pd.isna(s.succ_from):
                continue
            t_before = s.succ_from - pd.Timedelta(days=1)
            if not in_force([lg.valid_from[g]], [lg.valid_to[g]], t_before)[0]:
                continue  # the old article was not in force just before the change either
            items.append({"qid": int(q.qid), "query": q.query, "t_before": t_before,
                          "gold_before": {g}, "gold_after": set(lg.rows_of_aid[s.succ_aid]),
                          "old_doc": lg.doc_ids[g], "new_doc": s.succ_doc})
            break
        if len(items) >= args.max_questions:
            break
    print(f"{len(items)} time-travel questions", flush=True)

    # ---------------- run
    q_emb = lg.encode_queries([it["query"] for it in items])
    ti, _ = lg.retrieve(q_emb, k=args.k_retrieve * 3)
    res = {name: {"before": [], "after": []} for name in ["static", "extended_no_time", "lexgraph"]}
    for j, it in enumerate(tqdm(items, desc="time-travel", mininterval=30)):
        cache = {}

        def ce(rows):
            need = [r for r in rows if r not in cache]
            if need:
                for r, sc in zip(need, lg.ce_scores(it["query"], np.array(need))):
                    cache[r] = sc
            return np.array([cache[r] for r in rows])

        top_all = ti[j][:args.k_retrieve]
        top_qa = ti[j][is_qa[ti[j]]][:args.k_retrieve]
        for when, t in [("before", it["t_before"]), ("after", t_after)]:
            gold = it["gold_before"] if when == "before" else it["gold_after"]
            for name in res:
                if name == "static":
                    rows = np.array(top_qa[:args.k_rerank]); final = ce(list(rows))
                elif name == "extended_no_time":
                    rows = np.array(top_all[:args.k_rerank]); final = ce(list(rows))
                else:
                    rows = lg.candidates(q_emb[j], top_all, t, w.tau, args.k_rerank, True)
                    base = ce(list(rows))
                    stale, _, _ = lg.status(rows, t, w.tau)
                    prop = lg.propagation(rows, base) if w.mu else np.zeros(len(rows))
                    final = lg.combine(rows, base, stale, prop, w)
                o = np.argsort(-final, kind="stable")
                order = rows[o]
                st, _, _ = lg.status(order[:10], t, w.tau)
                r = ranks(list(order), gold)
                res[name][when].append({"rank": r, "stale10": float(st.mean())})

    summary = {"n_questions": len(items),
               "date_gap_days_median": float(np.median([(t_after - it["t_before"]).days for it in items])) if items else None}
    for name, d in res.items():
        row = {}
        for when in ["before", "after"]:
            rk = [x["rank"] for x in d[when]]
            row[when] = {"Hit@1": round(float(np.mean([r == 1 for r in rk])), 4),
                         "Hit@10": round(float(np.mean([r is not None for r in rk])), 4),
                         "MRR@10": round(float(np.mean([1 / r if r else 0 for r in rk])), 4),
                         "Stale@10": round(float(np.mean([x["stale10"] for x in d[when]])), 4)}
        row["correct_at_both_dates@10"] = round(float(np.mean(
            [(a["rank"] is not None) and (b["rank"] is not None) for a, b in zip(d["before"], d["after"])])), 4)
        summary[name] = row
    (out / "timetravel.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
