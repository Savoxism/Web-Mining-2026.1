"""Version alignment v2: which newer article supersedes an old one?

v1 accepted the most similar successor article when cosine >= 0.80 (64% judged correct). v2:
  1. features for every candidate pair (old passage, successor article): bi-encoder cosine, rank, cross-encoder score
     (fine-tuned reranker, old text as query), heading overlap, article-number distance, fine-amount overlap, length
     ratio, document types, direct relation type between the documents, and whether the successor document contains a
     clause-level amendment operation on the old article (from scripts/kg/build_amendments.py);
  2. labels from a judge LLM (gpt-oss-120b) on a stratified sample; train / validation / test split by old document;
  3. a gradient-boosting classifier, then one-to-one matching (Hungarian) between old and new articles of each
     (old document, successor document) pair; a match is accepted when its probability >= the validation threshold.
Writes data/kg/temporal/successors_v2.parquet (adds `prob`, `accept`) and a metrics JSON comparing v1 and v2.

Usage (needs the judge server):
    .venv/bin/python scripts/kg/align_versions.py --ce outputs/models/reranker-legal/final --judge-url http://127.0.0.1:8200/v1
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from scipy.optimize import linear_sum_assignment
from sentence_transformers import CrossEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lexgraph.temporal import is_partial  # noqa: E402

JUDGE = """You are checking a legal version alignment between Vietnamese legal articles. Reply with a JSON object only.
Old article:
{old}

Proposed newer article (from a later document that amends, replaces or repeals the old one):
{new}

Return {{"same_provision": 0 or 1}}: 1 if the newer article regulates the same legal matter as the old one, so that it
updates, replaces or amends it (the wording may differ); 0 otherwise."""

TOK = re.compile(r"\w+", re.UNICODE)
MONEY = re.compile(r"(\d{1,3}(?:\.\d{3})+)\s*đồng")
ART = re.compile(r"Điều\s+(\d+)")


def toks(s):
    return set(TOK.findall(str(s).lower()))


def jacc(a, b):
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal", default="data/kg/temporal")
    ap.add_argument("--kg", default="data/kg")
    ap.add_argument("--ce", required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument("--n-label", type=int, default=2400)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    td, kg = Path(args.temporal), Path(args.kg)
    rng = np.random.default_rng(args.seed)

    S = pd.read_parquet(td / "successors.parquet")
    C = pd.read_parquet(td / "corpus_ext.parquet", columns=["cid", "text", "aid", "doc_id", "loai_van_ban"])
    text_of_cid = dict(zip(C.cid, C.text))
    rows_of_aid = C.groupby("aid").text.apply(lambda x: "\n".join(x.tolist()[:3])).to_dict()
    dtype = pd.read_parquet(td / "docs_temporal.parquet").set_index("id").loai_van_ban
    E = pd.read_parquet(kg / "edges.parquet")
    E = E.assign(type=np.where(is_partial(E), "partially_repeals", E.type))  # feature keeps full / partial apart
    pair_type = E[E.type.isin(["amends", "replaces", "repeals", "partially_repeals"])].groupby(["src", "dst"]).type.apply(set).to_dict()
    ops = pd.read_parquet(kg / "amendments.parquet", columns=["amend_doc", "target_aid"])
    op_pairs = set(zip(ops.amend_doc, ops.target_aid))

    S["old_text"] = S.cid.map(text_of_cid).fillna("")
    S["new_text"] = S.succ_aid.map(rows_of_aid).fillna("")
    S = S[(S.old_text != "") & (S.new_text != "")].reset_index(drop=True)
    print(f"{len(S)} candidate pairs", flush=True)

    # ---------------- features
    ce = CrossEncoder(args.ce, device="cuda", max_length=512)
    ce.model.half()
    S["ce"] = ce.predict(list(zip(S.old_text.str[:1500], S.new_text.str[:1500])), batch_size=128,
                         show_progress_bar=True, activation_fn=torch.nn.Identity())
    head = lambda t: t.strip().strip('"“').split("\n")[0]  # noqa: E731
    S["head_jacc"] = [jacc(toks(head(a)), toks(head(b))) for a, b in zip(S.old_text, S.new_text)]
    S["tok_jacc"] = [jacc(toks(a[:3000]), toks(b[:3000])) for a, b in zip(S.old_text, S.new_text)]

    def art_no(aid):
        m = re.search(r"#(\d+)", str(aid))
        return int(m.group(1)) if m else -1
    S["art_dist"] = [abs(art_no(a) - art_no(b)) if art_no(a) >= 0 and art_no(b) >= 0 else 99
                     for a, b in zip(S.aid, S.succ_aid)]
    S["money_jacc"] = [jacc(set(MONEY.findall(a)), set(MONEY.findall(b))) for a, b in zip(S.old_text, S.new_text)]
    S["has_money"] = [int(bool(MONEY.search(a))) for a in S.old_text]
    S["len_ratio"] = [min(len(a), len(b)) / max(1, max(len(a), len(b))) for a, b in zip(S.old_text, S.new_text)]
    S["same_type"] = (S.doc_id.map(dtype) == S.succ_doc.map(dtype)).astype(int)
    for ty in ["amends", "replaces", "repeals", "partially_repeals"]:
        S[f"rel_{ty}"] = [int(ty in pair_type.get((s, d), set())) for s, d in zip(S.succ_doc, S.doc_id)]
    S["clause_op"] = [int((s, a) in op_pairs) for s, a in zip(S.succ_doc, S.aid)]
    S["explicit_i"] = S.explicit.astype(int)
    feats = ["sim", "rank", "ce", "head_jacc", "tok_jacc", "art_dist", "money_jacc", "has_money", "len_ratio",
             "same_type", "rel_amends", "rel_replaces", "rel_repeals", "rel_partially_repeals", "clause_op", "explicit_i"]

    # ---------------- judge labels (stratified by rank and similarity)
    S["sim_bin"] = pd.cut(S.sim, [0, 0.6, 0.7, 0.8, 0.9, 1.01], labels=False)
    per = max(1, args.n_label // (3 * 5))
    lab = S.sample(frac=1.0, random_state=args.seed).groupby(["rank", "sim_bin"]).head(per).copy()
    client = OpenAI(base_url=args.judge_url, api_key="EMPTY")

    def judge(i):
        r = lab.loc[i]
        for _ in range(3):
            try:
                out = client.chat.completions.create(
                    model=args.judge_model, temperature=0.0, max_tokens=300,
                    messages=[{"role": "user", "content": JUDGE.format(old=r.old_text[:2500], new=r.new_text[:3000])}],
                    extra_body={"reasoning_effort": "low"}).choices[0].message.content or ""
                m = re.search(r'"same_provision"\s*:\s*([01])', out)
                if m:
                    return int(m.group(1))
            except Exception:
                pass
        return -1

    with ThreadPoolExecutor(args.workers) as ex:
        lab["label"] = list(ex.map(judge, lab.index))
    lab = lab[lab.label >= 0]
    print(f"{len(lab)} judged pairs, positive rate {lab.label.mean():.3f}", flush=True)

    # split by old document so no document leaks across splits
    docs = lab.doc_id.unique()
    rng.shuffle(docs)
    n = len(docs)
    split = {d: ("test" if k < 0.2 * n else "val" if k < 0.35 * n else "train") for k, d in enumerate(docs)}
    lab["split"] = lab.doc_id.map(split)
    tr, va, te = (lab[lab.split == x] for x in ["train", "val", "test"])

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=args.seed)
    clf.fit(tr[feats], tr.label)
    pv = clf.predict_proba(va[feats])[:, 1]
    ths = np.linspace(0.2, 0.8, 25)
    acc_v = [((pv >= th).astype(int) == va.label).mean() for th in ths]
    thr = float(ths[int(np.argmax(acc_v))])

    def pr(y, yhat):
        tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        return {"accuracy": round(float((y == yhat).mean()), 4), "precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / max(1e-9, p + r), 4)}

    y = te.label.to_numpy()
    rule_v1 = ((te.sim >= 0.80) | te.explicit).astype(int).to_numpy() * (te["rank"] == 0).astype(int).to_numpy()
    p_te = clf.predict_proba(te[feats])[:, 1]

    # ---------------- apply to all pairs + Hungarian per (old doc, successor doc)
    S["prob"] = clf.predict_proba(S[feats])[:, 1]
    S["accept"] = False
    for _, g in S.groupby(["doc_id", "succ_doc"]):
        olds, news = sorted(g.aid.unique()), sorted(g.succ_aid.unique())
        oi, ni = {a: i for i, a in enumerate(olds)}, {a: i for i, a in enumerate(news)}
        M = np.full((len(olds), len(news)), -1e3)
        best = g.groupby(["aid", "succ_aid"]).prob.max()
        for (a, b), p in best.items():
            M[oi[a], ni[b]] = np.log(max(p, 1e-6))
        r_ind, c_ind = linear_sum_assignment(-M)
        for r_, c_ in zip(r_ind, c_ind):
            if M[r_, c_] > -1e3 and np.exp(M[r_, c_]) >= thr:
                S.loc[(g.index[(g.aid == olds[r_]) & (g.succ_aid == news[c_])]), "accept"] = True
    acc_map = S.set_index(["cid", "succ_aid"]).accept.to_dict()
    hung_te = np.array([int(acc_map.get((c, s), False)) for c, s in zip(te.cid, te.succ_aid)])

    metrics = {
        "candidate_pairs": len(S), "judged_pairs": len(lab), "positive_rate": round(float(lab.label.mean()), 4),
        "split_sizes": {k: int((lab.split == k).sum()) for k in ["train", "val", "test"]}, "threshold": thr,
        "test": {"v1_rule (rank0, cos>=0.80 or explicit)": pr(y, rule_v1),
                 "v2_classifier": pr(y, (p_te >= thr).astype(int)),
                 "v2_classifier_hungarian": pr(y, hung_te)},
        "accepted_v1": int((((S.sim >= 0.80) | S.explicit) & (S["rank"] == 0)).sum()),
        "accepted_v2": int(S.accept.sum()),
        "feature_importance_note": "HistGradientBoosting has no built-in importances; see permutation importance below",
    }
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(clf, te[feats], te.label, n_repeats=5, random_state=args.seed)
    metrics["permutation_importance"] = {f: round(float(v), 4) for f, v in sorted(zip(feats, pi.importances_mean), key=lambda x: -x[1])}
    keep = ["cid", "aid", "doc_id", "succ_aid", "succ_doc", "sim", "rank", "explicit", "prob", "accept", "clause_op", "ce"]
    S[keep].to_parquet(td / "successors_v2.parquet", index=False)
    lab.drop(columns=["old_text", "new_text"]).to_parquet(td / "alignment_labels.parquet", index=False)
    (td / "alignment_v2.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
