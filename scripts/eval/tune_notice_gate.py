"""Tune the change-notice gate on validation questions (no generation needed: a notice is written iff a verified
change fact reaches the context, enforced by lexgraph.llm.verify_notice).

Questions as in eval_answers.py, but from the validation split:
  * stable   (gold in force at t): any fact is a false alarm;
  * temporal (gold superseded):    the judge decides whether the answer really changed; a fact is expected then.
For every fact we keep CE(q, old clause) and CE(q, new clause); the gate keeps a fact when
CE(q, new) >= CE(q, old) - delta. Picks the delta with the highest recall at a false-alarm rate <= --max-fa.

Usage: LEXGRAPH_NOTICE_DELTA=inf .venv/bin/python scripts/eval/tune_notice_gate.py --bi ... --ce ... --weights ... --out ...
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexgraph.llm as L  # noqa: E402
from eval_answers import JUDGE_CHANGED, chat, parse_json  # noqa: E402
from lexgraph.pipeline import LexGraph, Weights  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", required=True)
    ap.add_argument("--ce", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--t-eval", default="2026-11-23")
    ap.add_argument("--n-stable", type=int, default=300)
    ap.add_argument("--n-temporal", type=int, default=150)
    ap.add_argument("--max-fa", type=float, default=0.10)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--llm-model", default="qwen3-4b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L.NOTICE_DELTA = float("inf")  # collect every fact, gate offline
    L.NOTICE_JACCARD = 1.01      # any wording difference counts; the Jaccard threshold is tuned offline
    t = pd.Timestamp(args.t_eval)
    best = json.loads(Path(args.weights).read_text())["best_val"]
    w = Weights(beta=best["beta"], gamma=best["gamma"], mu=best["mu"], inject=True)
    lg = LexGraph(args.bi, args.ce)
    from lexgraph.versions import VersionIndex
    lg.versions = VersionIndex("data/kg/amendments.parquet")  # as in the demo and eval_answers
    judge = OpenAI(base_url=args.judge_url, api_key="EMPTY")

    qs = pd.read_parquet(Path(args.data) / f"queries_{args.split}.parquet").sample(frac=1.0, random_state=7)
    stable, temporal = [], []
    for q in qs.itertuples(index=False):
        gold = sorted(lg.row_of[c] for c in q.cids if c in lg.row_of)
        if not gold or lg.doc_ids[gold[0]] == "":
            continue
        st, _, su = lg.status(np.array(gold), t, w.tau)
        if st.any() and any(s is not None and lg.rows_of_aid.get(s.succ_aid) for s in su):
            if len(temporal) < args.n_temporal:
                cur = set()
                for g, a, s in zip(gold, st, su):
                    cur |= set(lg.rows_of_aid[s.succ_aid]) if (a and s is not None and lg.rows_of_aid.get(s.succ_aid)) else {g}
                temporal.append({"query": q.query, "gold": gold, "cur": sorted(cur), "kind": "temporal"})
        elif not st.any() and len(stable) < args.n_stable:
            stable.append({"query": q.query, "gold": gold, "cur": gold, "kind": "stable"})
        if len(stable) >= args.n_stable and len(temporal) >= args.n_temporal:
            break
    items = stable + temporal
    print(f"{len(stable)} stable, {len(temporal)} temporal ({args.split})", flush=True)

    keys = ["old_doc", "old_art", "new_doc", "new_art", "old_ce", "new_ce", "money_changed", "jaccard", "old_clause", "new_clause"]
    for it in items:
        res = lg.search(it["query"], t, w, k_rerank=30, top=20)
        q_exp = L.expand_query(it["query"])
        q_vec = lg.encode_queries([q_exp])[0]
        rows = res.to_dict("records")
        # same candidate rule as lexgraph.llm.build_context: every old version that could yield the (single) fact
        valid = [r for r in rows if not r["stale"]][:5]
        aids = {L._s(r.get("aid")) for r in valid}
        top_ce = max([r["ce_score"] for r in valid], default=0.0)
        it["facts"] = []
        for r in rows:
            if r["stale"] and L._s(r.get("successor_aid")) and (r["successor_aid"] in aids or r["ce_score"] >= top_ce - 1.5):
                f = L.change_fact(lg, r, q_vec, q_exp, t)
                if f is not None:
                    it["facts"].append({k: f[k] for k in keys})
        # reverse candidates: articles in force in the context that replaced an older one (any age; filtered offline)
        it["rev_facts"] = []
        for r in valid[:3]:
            for old in L.predecessors(lg, L._s(r.get("aid")), t, 365 * 50):
                f = L.change_fact(lg, old, q_vec, q_exp, t)
                if f is not None:
                    it["rev_facts"].append({k: f[k] for k in keys} | {"age": int((t - old["_since"]).days)})

    def jchanged(it):
        old = "\n\n".join(str(lg.corpus.text.iat[r])[:2500] for r in it["gold"][:2])
        ref = "\n\n".join(str(lg.corpus.text.iat[r])[:2500] for r in it["cur"][:2])
        txt = chat(judge, args.judge_model, JUDGE_CHANGED.format(q=it["query"], old=old, new=ref, t=args.t_eval), max_tokens=256)
        it["changed"] = parse_json(txt, ["changed"])["changed"]

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(jchanged, temporal))

    # LLM check of every collected fact (Qwen3-4B, the answering model), with and without thinking
    llm = OpenAI(base_url=args.llm_url, api_key="EMPTY")
    facts = [(it, f) for it in items for f in it["facts"] + it["rev_facts"]]

    def chk(job):
        (it, f), th = job
        f[f"llm_{'think' if th else 'fast'}"] = int(L.llm_says_changed(llm, args.llm_model, L.expand_query(it["query"]),
                                                                    f["old_clause"], f["new_clause"], th))

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(chk, [(x, th) for x in facts for th in (False, True)]))
    json.dump({"n_items": len(items), "n_rev": sum(len(it["rev_facts"]) for it in items)}, sys.stdout); print()

    def rates(theta, delta=float("inf"), llm_mode=None, rev_days=0):
        def ok(f):
            return (f["money_changed"] or f["jaccard"] < theta) and f["new_ce"] >= f["old_ce"] - delta \
                and (llm_mode is None or f[f"llm_{llm_mode}"])

        def keep(it):  # as in build_context: forward facts first, reverse ones only when none passes
            if any(ok(f) for f in it["facts"]):
                return True
            return rev_days > 0 and any(ok(f) and f["age"] <= rev_days for f in it["rev_facts"])
        fa = float(np.mean([keep(it) for it in stable]))
        ch = [it for it in temporal if it["changed"]]
        rec = float(np.mean([keep(it) for it in ch])) if ch else 0.0
        un = [it for it in temporal if not it["changed"]]
        return {"jaccard_below": theta, "delta": delta, "llm_check": llm_mode, "rev_days": rev_days, "false_alarm_stable": round(fa, 4), "recall_changed": round(rec, 4),
                "notice_on_unchanged_temporal": round(float(np.mean([keep(it) for it in un])) if un else 0.0, 4)}

    grid = [rates(th, d, m, rd) for th in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01] for d in [float("inf"), 4]
            for m in [None, "fast", "think"] for rd in [0, 365, 730, 1825, 3650, 365 * 50]]
    ok = [g for g in grid if g["false_alarm_stable"] <= args.max_fa]
    pick = max(ok, key=lambda g: (g["recall_changed"], -g["false_alarm_stable"])) if ok else \
        min(grid, key=lambda g: g["false_alarm_stable"] - g["recall_changed"])
    res = {"split": args.split, "n_stable": len(stable), "n_temporal": len(temporal),
           "n_temporal_changed": int(sum(it["changed"] for it in temporal)), "grid": grid, "chosen": pick}
    (out / "notice_gate.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
