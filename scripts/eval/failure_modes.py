"""Failure modes of the time-aware answers (from eval_answers.py output, no new generation).

Every in-scope question the judge marks incorrect is assigned the first cause that applies:
  * retrieval miss        - no source shown to the LLM is (a chunk of) a current gold article;
  * outdated source cited - a current gold article is shown, but the answer cites only stale sources;
  * wrong refusal         - a current gold article is shown, but the answer refuses;
  * misread / incomplete  - a current gold article is shown and cited (or nothing is cited), the answer is still wrong.
Change-notice errors are counted separately: false notices on stable questions and missed notices on questions whose
answer really changed (judge), split by whether a verified change fact reached the context.

Usage: .venv/bin/python scripts/eval/failure_modes.py --answers <eval_answers metrics dir>/answers.json --out <json>
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

FACTS_HEADER = "THAY ĐỔI ĐÃ XÁC MINH"
SUB = """A Vietnamese legal assistant answered a question; the reference article (law in force) was among its sources, yet
the answer was graded incorrect. Classify the main error. Reply with a JSON object only.
Question: {q}

Reference article:
{ref}

Answer:
{ans}

Return {{"error": "incomplete" | "wrong_clause_or_number" | "unsupported_claim" | "actually_correct"}}:
incomplete = misses an essential part of the rule; wrong_clause_or_number = uses another clause, point, amount, time limit
or condition than the reference; unsupported_claim = adds legal content not in the reference; actually_correct = the
answer agrees with the reference."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--corpus", default="data/kg/temporal/corpus_ext.parquet")
    ap.add_argument("--judge-url", default=None, help="sub-classify 'misread / incomplete' with this judge")
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A = json.loads(Path(args.answers).read_text())
    corpus = pd.read_parquet(args.corpus, columns=["aid", "text"]).reset_index(drop=True)
    aid_of_row = corpus.aid.to_dict()

    def has_notice(a):
        return ("thay đổi pháp luật" in a.lower()) or ("⚠" in a)

    causes, examples = Counter(), {}
    n_wrong = 0
    for it in A:
        if it["kind"] == "out_of_scope" or it["judge_ours"]["correct"]:
            continue
        n_wrong += 1
        cur_aids = {aid_of_row.get(r) for r in it["cur"]} - {None}
        src = it["src_ours"]
        gold_k = {s["k"] for s in src if s.get("aid") in cur_aids}
        cited = {int(x) for x in re.findall(r"\[(\d+)\]", it["ans_ours"])}
        stale_k = {s["k"] for s in src if s.get("stale")}
        if not gold_k:
            c = "retrieval miss"
        elif "không tìm thấy đủ căn cứ" in it["ans_ours"].lower():
            c = "wrong refusal"
        elif cited and cited <= stale_k:
            c = "outdated source cited"
        else:
            c = "misread / incomplete"
        causes[c] += 1
        examples.setdefault(c, []).append({"query": it["query"], "kind": it["kind"], "answer": it["ans_ours"][:300],
                                           "ref": "\n\n".join(str(corpus.text.iat[r])[:2500] for r in it["cur"][:2]),
                                           "full_answer": it["ans_ours"]})

    sub = Counter()
    if args.judge_url and examples.get("misread / incomplete"):
        from concurrent.futures import ThreadPoolExecutor
        from openai import OpenAI
        client = OpenAI(base_url=args.judge_url, api_key="EMPTY")

        def judge(e):
            for _ in range(3):
                try:
                    out = client.chat.completions.create(
                        model=args.judge_model, temperature=0.0, max_tokens=400,
                        messages=[{"role": "user", "content": SUB.format(q=e["query"], ref=e["ref"], ans=e["full_answer"][:3000])}],
                        extra_body={"reasoning_effort": "low"}).choices[0].message.content or ""
                    m = re.search(r'"error"\s*:\s*"(\w+)"', out)
                    if m:
                        return m.group(1)
                except Exception:
                    pass
            return "unknown"

        with ThreadPoolExecutor(16) as ex:
            sub = Counter(ex.map(judge, examples["misread / incomplete"]))
    for ex in examples.values():
        for e in ex:
            e.pop("ref"); e.pop("full_answer")

    stable = [it for it in A if it["kind"] == "stable"]
    changed = [it for it in A if it["kind"] == "temporal" and it.get("changed")]
    false_n = [it for it in stable if has_notice(it["ans_ours"])]
    missed = [it for it in changed if not has_notice(it["ans_ours"])]
    res = {
        "in_scope": sum(it["kind"] != "out_of_scope" for it in A), "wrong": n_wrong,
        "causes": dict(causes.most_common()), "misread_subtypes": dict(sub.most_common()),
        "causes_by_kind": {k: dict(Counter(c for c, ex in examples.items() for e in ex if e["kind"] == k))
                           for k in ["stable", "temporal"]},
        "false_notices": len(false_n), "stable": len(stable),
        "missed_notices": len(missed), "changed": len(changed),
        "missed_without_fact": sum(FACTS_HEADER not in it["ctx_ours"] for it in missed),
        "examples": {c: ex[:3] for c, ex in examples.items()},
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
