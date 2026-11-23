"""Fine-tune a cross-encoder re-ranker with BCE on (query, positive) and semi-hard negatives.

Launch on all GPUs (DDP):
    torchrun --nproc_per_node 2 scripts/train/train_crossencoder.py --base BAAI/bge-reranker-v2-m3 \
        --revision <sha> --train data/train/ce_train.parquet --val-candidates data/train/ce_val_candidates.parquet \
        --out outputs/models/reranker-legal
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val-candidates", required=True)
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-neg", type=int, default=7)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--val-queries", type=int, default=1000)
    ap.add_argument("--val-depth", type=int, default=30)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (saves activation memory)")
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    model = CrossEncoder(args.base, revision=args.revision, num_labels=1, max_length=args.max_length)

    df = pd.read_parquet(args.train)
    q, p, y = [], [], []
    for r in df.itertuples(index=False):
        q.append(r.anchor); p.append(r.positive); y.append(1.0)
        for k in range(1, args.n_neg + 1):
            q.append(r.anchor); p.append(getattr(r, f"negative_{k}")); y.append(0.0)
    train_ds = Dataset.from_dict({"query": q, "passage": p, "label": y}).shuffle(seed=args.seed)

    corpus = pd.read_parquet(Path(args.data) / "corpus.parquet").set_index("cid").text
    vc = pd.read_parquet(args.val_candidates).sample(args.val_queries, random_state=args.seed)
    samples = [{"query": r.query, "positive": [corpus[c] for c in r.cids if c in corpus.index],
                "documents": [corpus[c] for c in list(r.cand_cids)[:args.val_depth]]} for r in vc.itertuples()]
    evaluator = CrossEncoderRerankingEvaluator(samples=samples, at_k=10, name="val", batch_size=64,
                                               always_rerank_positives=False, show_progress_bar=False)

    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(float(args.n_neg)))
    targs = CrossEncoderTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_ndcg@10",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.grad_ckpt,
    )
    if rank == 0:
        info = {"world_size": world, "global_batch": world * args.per_device_batch * args.grad_accum,
                "train_pairs": len(train_ds), "val_samples": len(samples), **vars(args)}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "train_config.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2), flush=True)
        print("zero-shot val:", evaluator(model), flush=True)

    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator)
    trainer.train()
    if rank == 0:
        final = Path(args.out) / "final"
        model.save_pretrained(str(final))
        metrics = evaluator(model)
        (final / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("final val:", metrics, flush=True)


if __name__ == "__main__":
    main()
