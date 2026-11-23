"""Second LLM judge for the saved answers of eval_answers.py, and agreement with the first judge.

Re-grades every in-scope answer (correct, faithful) and every changed question (did the answer really change?) with the
same prompts as eval_answers.py, using a second model (Qwen3-32B by default, thinking mode), then reports the metrics
under the second judge and the agreement (raw and Cohen's kappa) with the first judge (gpt-oss-120b).

Usage: .venv/bin/python scripts/eval/rejudge_answers.py --answers <dir>/answers.json --judge-url http://127.0.0.1:8300/v1
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_answers import JUDGE_CHANGED, JUDGE_QA, parse_json  # noqa: E402

SETTINGS = ["none", "plain", "ours"]


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--corpus", default="data/kg/temporal/corpus_ext.parquet")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8300/v1")
    ap.add_argument("--judge-model", default="qwen3-32b")
    ap.add_argument("--t-eval", default="2026-11-23")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A = json.loads(Path(args.answers).read_text())
    text = pd.read_parquet(args.corpus, columns=["text"]).reset_index(drop=True).text
    client = OpenAI(base_url=args.judge_url, api_key="EMPTY")

    def chat(prompt):
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.judge_model, temperature=0.0, max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}})
                return (r.choices[0].message.content or "").split("</think>")[-1]
            except Exception:
                pass
        return ""

    def ref(rows):
        return "\n\n".join(str(text.iat[r])[:2500] for r in rows[:2]) or "(none)"

    in_scope = [it for it in A if it["kind"] != "out_of_scope"]

    def jqa(job):
        it, s = job
        ctx = it.get(f"ctx_{s}", "(no sources)") if s != "none" else "(no sources)"
        out = chat(JUDGE_QA.format(q=it["query"], t=args.t_eval, ref=ref(it["cur"]), ctx=ctx[:9000], ans=it[f"ans_{s}"][:4000]))
        it[f"judge2_{s}"] = parse_json(out, ["correct", "faithful"])

    def jch(it):
        out = chat(JUDGE_CHANGED.format(q=it["query"], old=ref(it["gold"]), new=ref(it["cur"]), t=args.t_eval))
        it["changed2"] = parse_json(out, ["changed"])["changed"]

    temporal = [it for it in A if it["kind"] == "temporal"]
    from tqdm import tqdm
    jobs = [(jqa, (it, s)) for it in in_scope for s in SETTINGS] + [(jch, it) for it in temporal]
    with ThreadPoolExecutor(args.workers) as ex:
        for _ in tqdm(ex.map(lambda j: j[0](j[1]), jobs), total=len(jobs), desc="rejudge", mininterval=5):
            pass

    def has_notice(a):
        return ("thay đổi pháp luật" in a.lower()) or ("⚠" in a)

    res = {"judge": args.judge_model, "settings": {}, "agreement": {}}
    for s in SETTINGS:
        stb = [it for it in in_scope if it["kind"] == "stable"]
        tmp = [it for it in in_scope if it["kind"] == "temporal"]
        ch = [it for it in tmp if it["changed2"]]
        res["settings"][s] = {
            "correct": round(float(np.mean([it[f"judge2_{s}"]["correct"] for it in in_scope])), 3),
            "correct_temporal": round(float(np.mean([it[f"judge2_{s}"]["correct"] for it in tmp])), 3),
            "faithful": round(float(np.mean([it[f"judge2_{s}"]["faithful"] for it in in_scope])), 3),
            "notice_recall_changed": round(float(np.mean([has_notice(it[f"ans_{s}"]) for it in ch] or [0])), 3),
            "false_notice_stable": round(float(np.mean([has_notice(it[f"ans_{s}"]) for it in stb])), 3),
        }
    for key in ["correct", "faithful"]:
        a = [it[f"judge_{s}"][key] for it in in_scope for s in SETTINGS]
        b = [it[f"judge2_{s}"][key] for it in in_scope for s in SETTINGS]
        po, k = kappa(a, b)
        res["agreement"][key] = {"raw": round(po, 3), "kappa": round(k, 3), "n": len(a)}
    a = [it["changed"] for it in temporal]
    b = [it["changed2"] for it in temporal]
    po, k = kappa(a, b)
    res["agreement"]["changed"] = {"raw": round(po, 3), "kappa": round(k, 3), "n": len(a)}
    res["n_changed2"] = int(sum(b))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
