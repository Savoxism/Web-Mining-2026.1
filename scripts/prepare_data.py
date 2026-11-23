"""EDA, corpus construction and train/val/test split (by qid) for the legal QA dataset.

Usage:
    uv run python scripts/prepare_data.py \
        --input data/soict-hackathon-2025-legal-corpus/train-00000-of-00001.parquet \
        --out data/processed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def describe_lengths(s: pd.Series) -> dict:
    words = s.fillna("").str.split().str.len()
    q = words.quantile([0.5, 0.9, 0.99]).tolist()
    return {
        "mean_words": round(float(words.mean()), 1),
        "median_words": q[0],
        "p90_words": q[1],
        "p99_words": q[2],
        "max_words": int(words.max()),
        "empty": int((words == 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    stats: dict = {"rows": len(df), "columns": list(df.columns)}

    # ---------- EDA ----------
    stats["unique_qid"] = int(df.qid.nunique())
    stats["unique_cid"] = int(df.cid.nunique())
    stats["unique_query_text"] = int(df["query"].nunique())
    stats["unique_pos_text"] = int(df.pos.nunique())
    stats["duplicate_rows"] = int(df.duplicated(["qid", "cid"]).sum())

    pos_per_q = df.groupby("qid").cid.nunique()
    stats["positives_per_query"] = {str(k): int(v) for k, v in pos_per_q.value_counts().sort_index().items()}
    q_per_cid = df.groupby("cid").qid.nunique()
    stats["queries_per_passage"] = {
        "mean": round(float(q_per_cid.mean()), 2),
        "max": int(q_per_cid.max()),
        "share_with_1": round(float((q_per_cid == 1).mean()), 3),
    }
    texts_per_cid = df.groupby("cid").pos.nunique()
    stats["cids_with_multiple_texts"] = int((texts_per_cid > 1).sum())

    stats["query_length"] = describe_lengths(df["query"])
    stats["passage_length"] = describe_lengths(df.drop_duplicates("cid").pos)
    if "prompt" in df.columns:
        stats["prompt_length"] = describe_lengths(df.prompt)
        stats["unique_prompt"] = int(df.prompt.nunique())

    # ---------- Corpus: one text per cid (longest variant) ----------
    corpus = (
        df.assign(_len=df.pos.str.len())
        .sort_values("_len", ascending=False)
        .drop_duplicates("cid")[["cid", "pos"]]
        .rename(columns={"pos": "text"})
        .sort_values("cid")
        .reset_index(drop=True)
    )
    corpus.to_parquet(out / "corpus.parquet", index=False)
    stats["corpus_size"] = len(corpus)

    # ---------- Queries with all relevant cids ----------
    queries = (
        df.drop_duplicates(["qid", "cid"])
        .groupby("qid")
        .agg(query=("query", "first"), cids=("cid", lambda x: sorted(set(int(c) for c in x))))
        .reset_index()
    )

    # ---------- Split by qid ----------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(queries))
    n_test = int(len(queries) * args.test_frac)
    n_val = int(len(queries) * args.val_frac)
    split = np.full(len(queries), "train", dtype=object)
    split[perm[:n_test]] = "test"
    split[perm[n_test : n_test + n_val]] = "val"
    queries["split"] = split

    for name in ["train", "val", "test"]:
        part = queries[queries.split == name].drop(columns="split").reset_index(drop=True)
        part.to_parquet(out / f"queries_{name}.parquet", index=False)
        stats[f"n_queries_{name}"] = len(part)

    # Leakage check: the same query text should not appear in two splits.
    by_text = queries.groupby("query").split.nunique()
    stats["query_texts_in_multiple_splits"] = int((by_text > 1).sum())

    # Passages that only appear in test (unseen during training).
    train_cids = set(c for cs in queries[queries.split == "train"].cids for c in cs)
    test_cids = [c for cs in queries[queries.split == "test"].cids for c in cs]
    stats["test_positives_unseen_in_train"] = round(float(np.mean([c not in train_cids for c in test_cids])), 3)

    # ---------- Samples ----------
    samples = []
    for _, r in df.sample(3, random_state=args.seed).iterrows():
        samples.append(
            {
                "qid": int(r.qid),
                "cid": int(r.cid),
                "query": r["query"],
                "pos": r.pos[:500],
                "prompt": (r.prompt[:1500] if isinstance(r.get("prompt"), str) else None),
            }
        )
    stats["samples"] = samples

    (out / "eda_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "samples"}, ensure_ascii=False, indent=2))
    print(f"\nSaved corpus, splits and eda_stats.json to {out}/")


if __name__ == "__main__":
    main()
