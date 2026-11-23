## Retrieval only (test, split v2, extended corpus)

| Bi-encoder | MRR@10 | R@10 | R@100 | MRR@10 cur | T-MRR@10 cur |
|---|---|---|---|---|---|
| bge-m3 zero-shot | 0.539 | 0.742 | 0.900 | 0.489 | 0.267 |
| bge-m3 fine-tuned | 0.650 | 0.865 | 0.972 | 0.584 | 0.311 |

## Pipeline, base models (bge-m3 + bge-reranker-v2-m3)
n_test = 11885, temporal subset = 2942; best val weights: beta=0.25, gamma=2.0, mu=0.0

| System | MRR@10 | R@10 | MRR@10 cur | Stale@10 | T-MRR@10 cur | T-Hit@10 cur | T-Stale@10 |
|---|---|---|---|---|---|---|---|
| retriever + re-ranker | 0.624 | 0.792 | 0.555 | 0.254 | 0.273 | 0.610 | 0.366 |
| + graph (beta, mu) | 0.628 | 0.792 | 0.555 | 0.256 | 0.269 | 0.607 | 0.369 |
| + time (gamma, injection) | 0.562 | 0.764 | 0.587 | 0.108 | 0.410 | 0.686 | 0.174 |
| full (graph + time) | 0.567 | 0.768 | 0.587 | 0.113 | 0.406 | 0.686 | 0.183 |
| full - PageRank | 0.562 | 0.764 | 0.587 | 0.108 | 0.410 | 0.686 | 0.174 |
| full - propagation | 0.567 | 0.768 | 0.587 | 0.113 | 0.406 | 0.686 | 0.183 |
| full - successor injection | 0.568 | 0.770 | 0.585 | 0.117 | 0.395 | 0.644 | 0.190 |
| full - explicit references | 0.568 | 0.768 | 0.587 | 0.111 | 0.405 | 0.683 | 0.178 |
| full, amends edges only | 0.576 | 0.773 | 0.580 | 0.105 | 0.381 | 0.657 | 0.163 |
| full, replaces/repeals only | 0.580 | 0.772 | 0.579 | 0.100 | 0.380 | 0.660 | 0.149 |

## Pipeline, fine-tuned models
n_test = 11885, temporal subset = 2942; best val weights: beta=0.25, gamma=2.0, mu=0.0

| System | MRR@10 | R@10 | MRR@10 cur | Stale@10 | T-MRR@10 cur | T-Hit@10 cur | T-Stale@10 |
|---|---|---|---|---|---|---|---|
| retriever + re-ranker | 0.704 | 0.885 | 0.630 | 0.261 | 0.334 | 0.664 | 0.375 |
| + graph (beta, mu) | 0.704 | 0.885 | 0.630 | 0.261 | 0.334 | 0.664 | 0.375 |
| + time (gamma, injection) | 0.662 | 0.872 | 0.659 | 0.141 | 0.440 | 0.723 | 0.228 |
| full (graph + time) | 0.664 | 0.874 | 0.658 | 0.145 | 0.435 | 0.719 | 0.237 |
| full - PageRank | 0.662 | 0.872 | 0.659 | 0.141 | 0.440 | 0.723 | 0.228 |
| full - propagation | 0.664 | 0.874 | 0.658 | 0.145 | 0.435 | 0.719 | 0.237 |
| full - successor injection | 0.664 | 0.874 | 0.657 | 0.147 | 0.431 | 0.699 | 0.240 |
| full - explicit references | 0.665 | 0.874 | 0.657 | 0.143 | 0.434 | 0.718 | 0.232 |
| full, amends edges only | 0.670 | 0.876 | 0.652 | 0.134 | 0.419 | 0.702 | 0.209 |
| full, replaces/repeals only | 0.673 | 0.877 | 0.651 | 0.129 | 0.416 | 0.704 | 0.197 |

## Training train_bi

GPUs 2, global batch 128, epochs 1.0, lr 1e-05; val_cosine_ndcg@10 = 0.7705, val_cosine_mrr@10 = 0.7409

## Training train_ce

GPUs 2, global batch 128, epochs 1.0, lr 1e-05; val_mrr@10 = 0.8081, val_ndcg@10 = 0.8250, val_base_mrr@10 = 0.7398, val_base_ndcg@10 = 0.7690

## Answer quality (judge gpt-oss-120b; 100 stable, 50 temporal, 20 out-of-scope)

| Setting | Citation acc. | Correct | Correct (stable) | Correct (temporal) | Faithful | Change notice (temporal) | False notice (stable) | Correct refusals | Over-refusals |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B, no retrieval | – | 0.100 | 0.090 | 0.120 | 0.007 | 0.000 | 0.000 | 0.550 | 0.027 |
| Qwen3-4B + plain RAG | 0.440 | 0.647 | 0.670 | 0.600 | 0.720 | 0.100 | 0.050 | 0.950 | 0.047 |
| Qwen3-4B + time-aware RAG (ours) | 0.560 | 0.713 | 0.700 | 0.740 | 0.680 | 0.940 | 0.260 | 0.950 | 0.020 |

## Version alignment audit: accuracy 0.640 on 100 pairs (explicit 0.286 on 7, similarity only 0.667)


## Time-travel benchmark (2,707 questions, asked the day before the change and on 2026-11-23)

| System | Hit@10 before | Hit@10 after | Hit@10 at both dates | Stale@10 after |
|---|---|---|---|---|
| static RAG (QA corpus) | 0.917 | 0.051 | 0.050 | 0.522 |
| extended corpus, no time signals | 0.855 | 0.602 | 0.569 | 0.366 |
| LexGraph-VN | 0.875 | 0.661 | 0.632 | 0.231 |
