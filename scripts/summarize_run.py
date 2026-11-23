"""Collect the metrics of run finetune_v1_20261123 into markdown tables (run-outputs/out_finetune_v1.md).

Usage: .venv/bin/python scripts/summarize_run.py --run outputs/runs/finetune_v1_20261123 --out run-outputs/out_finetune_v1.md
"""

import argparse
import json
from pathlib import Path


def f(x, nd=3):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/runs/finetune_v1_20261123")
    ap.add_argument("--out", default="run-outputs/out_finetune_v1.md")
    args = ap.parse_args()
    run = Path(args.run)
    lines = []

    # retrieval only
    rows = []
    for tag, name in [("eval_base", "bge-m3 zero-shot"), ("eval_ft", "bge-m3 fine-tuned")]:
        p = run / tag / "metrics" / "results.json"
        if p.exists():
            r = json.loads(p.read_text())
            ro = r["retrieval_only_test"]
            rows.append(f"| {name} | {f(ro['MRR@10'])} | {f(ro['R@10'])} | {f(ro['R@100'])} | {f(ro['MRR@10_cur'])} | {f(ro['T_MRR@10_cur'])} |")
    if rows:
        lines += ["## Retrieval only (test, split v2, extended corpus)", "",
                  "| Bi-encoder | MRR@10 | R@10 | R@100 | MRR@10 cur | T-MRR@10 cur |", "|---|---|---|---|---|---|", *rows, ""]

    # pipeline variants
    for tag, name in [("eval_base", "base models (bge-m3 + bge-reranker-v2-m3)"),
                      ("eval_ft", "fine-tuned models")]:
        p = run / tag / "metrics" / "results.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        b = r["best_val"]
        lines += [f"## Pipeline, {name}",
                  f"n_test = {r['n_test']}, temporal subset = {r['n_test_temporal']}; best val weights: "
                  f"beta={b['beta']}, gamma={b['gamma']}, mu={b['mu']}", "",
                  "| System | MRR@10 | R@10 | MRR@10 cur | Stale@10 | T-MRR@10 cur | T-Hit@10 cur | T-Stale@10 |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, v in r["variants"].items():
            lines.append(f"| {k} | {f(v['MRR@10'])} | {f(v['R@10'])} | {f(v['MRR@10_cur'])} | {f(v['Stale@10'])} | "
                         f"{f(v['T_MRR@10_cur'])} | {f(v['T_Hit@10_cur'])} | {f(v['T_Stale@10'])} |")
        lines.append("")

    # training
    for tag, model in [("train_bi", "outputs/models/bge-m3-legal"), ("train_ce", "outputs/models/reranker-legal")]:
        cfg = Path(model) / "train_config.json"
        vm = Path(model) / "final" / "val_metrics.json"
        if cfg.exists() and vm.exists():
            c = json.loads(cfg.read_text()); m = json.loads(vm.read_text())
            key = [k for k in m if k.endswith(("mrr@10", "ndcg@10"))]
            lines += [f"## Training {tag}", "",
                      f"GPUs {c['world_size']}, global batch {c['global_batch']}, epochs {c['epochs']}, lr {c['lr']}; "
                      + ", ".join(f"{k} = {m[k]:.4f}" for k in key), ""]

    # answers
    p = run / "eval_answers" / "metrics" / "metrics.json"
    if p.exists():
        m = json.loads(p.read_text())
        lines += [f"## Answer quality (judge gpt-oss-120b; {m['n_stable']} stable, {m['n_temporal']} temporal, "
                  f"{m['n_out_of_scope']} out-of-scope)", "",
                  "| Setting | Citation acc. | Correct | Correct (stable) | Correct (temporal) | Faithful | Change notice (temporal) | False notice (stable) | Correct refusals | Over-refusals |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for a in m["answers"]:
            lines.append(f"| {a['setting']} | {f(a['citation_acc'])} | {f(a['correct'])} | {f(a['correct_stable'])} | "
                         f"{f(a['correct_temporal'])} | {f(a['faithful'])} | {f(a['change_notice_temporal'])} | "
                         f"{f(a['change_notice_false_alarm_stable'])} | {f(a['correct_refusals_oos'])} | {f(a['over_refusals_in_scope'])} |")
        al = m["alignment_audit"]
        lines += ["", f"## Version alignment audit: accuracy {f(al['accuracy'])} on {al['n']} pairs "
                      f"(explicit {f(al['accuracy_explicit'])} on {al['n_explicit']}, similarity only {f(al['accuracy_similarity_only'])})", ""]

    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
