# Project
- name: Web-Mining-2026.1 (LexGraph-VN)
- run_id: finetune_v1_20261123
- local_path: /Users/savoxism/Documents/GitHub/Web-Mining-2026.1
- ssh_host: H200_ANNP36
- remote_path: /workspace/Web-Mining-2026.1
- git_remote: git@github.com:Savoxism/Web-Mining-2026.1.git (origin), branch main
- source_revision: commit that adds this file (recorded in run-outputs/out_finetune_v1.md)

# Data
- QA pairs: data/soict-hackathon-2025-legal-corpus/train-00000-of-00001.parquet (sha256 66e5a502…)
- split v2 (grouped by normalised question text, seed 42): data/processed_v2/{corpus,queries_train,queries_val,queries_test}.parquet
  - 95,231 / 11,891 / 11,885 questions, 68,663 passages
- KG: data/kg/ (built by scripts/kg/build_kg.py from th1nhng0/vietnamese-legal-documents @ 8977887f)
- temporal layer + extended corpus: data/kg/temporal/ (scripts/kg/build_temporal.py, base bge-m3, tau 0.80)
- bi-encoder training rows: data/train/biencoder_train.parquet (mined with base bge-m3)
- cross-encoder training rows: data/train/ce_train.parquet (mined with fine-tuned bge-m3)

# Model
- bi-encoder: BAAI/bge-m3 @ 5617a9f61b028005a4858fdac845db406aefb181
- cross-encoder: BAAI/bge-reranker-v2-m3 @ 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
- LLM: Qwen/Qwen3-4B @ 1cfa9a7208912126459214e8b04321603b3df60c (vLLM, demo + answer evaluation)

# Hyperparameters
- seed 42 everywhere; bf16; DDP on 2 GPUs (torchrun --nproc_per_node 2)
- negatives (bi-encoder): top-100 of base bge-m3, rank >= 3, score < sim(q, pos) - 0.05, 3 per pair
- bi-encoder: MultipleNegativesRankingLoss (gathered across devices), per-device batch 64, grad accum 1,
  global batch 128 queries; lr 1e-5 linear, warmup 0.1; 1 epoch; max_seq_len 512; eval every 200 steps on
  2,000 val questions (MRR@10), best checkpoint kept
- negatives (cross-encoder): top-100 of fine-tuned bge-m3, same rule, 7 per pair, 50,000 train questions
- cross-encoder: BCE with pos_weight 7, per-device batch 64 pairs, global 128 pairs; lr 1e-5 linear, warmup 0.1;
  1 epoch; max_length 512; eval every 500 steps on 1,000 val questions (top-30 of fine-tuned bge-m3, NDCG@10)
- pipeline evaluation: top-100 retrieval, top-30 re-ranked + injected successors, t_eval 2026-11-23,
  beta/gamma/mu grid-searched on 1,500 val questions by MRR@10 against the current gold
- GPU policy: phases that need one GPU run two at a time (one per GPU); training phases use both GPUs

# Outputs
- run dir: outputs/runs/finetune_v1_20261123/<phase>/ (log, exit_code, metrics)
- models: outputs/models/bge-m3-legal/final, outputs/models/reranker-legal/final
- evaluation: outputs/runs/finetune_v1_20261123/eval_{base,ft}/results.json
- report: run-outputs/out_finetune_v1.md

# History
- 2026-11-23: created (ACL run stopped by the user; both GPUs free).
- 2026-11-23 17:12 (giờ VN): train_bi failed at NCCL init (NVLS multicast "operation cannot be performed in the
  present state" under the CUDA MPS daemon). Old → new: environment only, NCCL_NVLS_ENABLE=0 added to
  scripts/phase.sh (same setting the ACL project uses). Data, seed, batch and hyperparameters unchanged; the failed
  log is kept in outputs/runs/finetune_v1_20261123/train_bi_failed_nvls/. Gradient checkpointing enabled for the
  bi-encoder (per-device batch 64 x 5 texts x 512 tokens), which changes memory, not the objective.
- 2026-11-23 17:25 (giờ VN): train_bi failed at step 1 (DDP: XLM-R pooler parameters 389/390 receive no gradient).
  Old → new: ddp_find_unused_parameters False → True, non-reentrant gradient checkpointing. Objective, data, seed and
  batch unchanged. Zero-shot val (2,000 questions) from that run: MRR@10 0.619, Recall@100 0.930. Failed log kept in
  train_bi_failed_ddp/.
