"""Fine-tune a bi-encoder with MultipleNegativesRankingLoss + mined semi-hard negatives.

Launch on all GPUs (DDP):
    torchrun --nproc_per_node 2 scripts/train/train_biencoder.py --base BAAI/bge-m3 --revision <sha> \
        --train data/train/biencoder_train.parquet --out outputs/models/bge-m3-legal
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--train", required=True)
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-neg", type=int, default=3)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=384)
    ap.add_argument("--val-queries", type=int, default=2000)
    ap.add_argument("--eval-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (saves activation memory)")
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    model = SentenceTransformer(args.base, revision=args.revision)
    model.max_seq_length = args.max_seq_len

    df = pd.read_parquet(args.train)
    cols = ["anchor", "positive"] + [f"negative_{k}" for k in range(1, args.n_neg + 1)]
    train_ds = Dataset.from_pandas(df[cols], preserve_index=False).shuffle(seed=args.seed)

    data = Path(args.data)
    corpus = pd.read_parquet(data / "corpus.parquet")
    val = pd.read_parquet(data / "queries_val.parquet").sample(args.val_queries, random_state=args.seed)
    evaluator = InformationRetrievalEvaluator(
        queries={str(q): t for q, t in zip(val.qid, val["query"])},
        corpus={str(c): t for c, t in zip(corpus.cid, corpus.text)},
        relevant_docs={str(q): {str(c) for c in cs} for q, cs in zip(val.qid, val.cids)},
        name="val", mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 10], precision_recall_at_k=[10, 100],
        map_at_k=[10], batch_size=256, show_progress_bar=False,
    )

    try:
        loss = losses.MultipleNegativesRankingLoss(model, gather_across_devices=True)
        gathered = True
    except TypeError:
        loss = losses.MultipleNegativesRankingLoss(model)
        gathered = False

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_cosine_mrr@10",
        greater_is_better=True,
        logging_steps=20,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.grad_ckpt,
    )
    if rank == 0:
        info = {"world_size": world, "per_device_batch": args.per_device_batch, "grad_accum": args.grad_accum,
                "global_batch": world * args.per_device_batch * args.grad_accum, "gather_across_devices": gathered,
                "train_rows": len(train_ds), **vars(args)}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "train_config.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2), flush=True)

    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator)
    if rank == 0:
        print("zero-shot val:", evaluator(model), flush=True)
    trainer.train()
    if rank == 0:
        final = Path(args.out) / "final"
        model.save_pretrained(str(final))
        metrics = evaluator(model)
        (final / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("final val:", metrics, flush=True)


if __name__ == "__main__":
    main()
