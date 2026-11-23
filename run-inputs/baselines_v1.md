# Project
- name: Web-Mining-2026.1 (LexGraph-VN)
- run_id: baselines_v1_20261123
- local_path: /Users/savoxism/Documents/GitHub/Web-Mining-2026.1
- ssh_host: H200_ANNP36
- remote_path: /workspace/Web-Mining-2026.1
- git_remote: git@github.com:Savoxism/Web-Mining-2026.1.git (origin)
- branch: main
- source_revision: commit that adds this file (recorded in run-outputs/out_baselines_v1.md)

# Data
- input: data/soict-hackathon-2025-legal-corpus/train-00000-of-00001.parquet
  - source: hf://datasets/Savoxism/soict-hackathon-2025-legal-corpus @ 6b7f1ceaa47c4915707229fb607469d01da6a59b
  - sha256: 66e5a50237943518834d2ff0045fba30aad631aacc607a0a555422033be2eb08
  - 132,424 rows; fields: query, cid, qid, pos, prompt
- processed (built by scripts/prepare_data.py, seed 42): data/processed/
  - corpus.parquet: one text per cid (longest variant)
  - queries_{train,val,test}.parquet: split by qid, 80/10/10
- benchmark: queries_test.parquet (retrieval over the full corpus)
- manifest: data/processed/eda_stats.json

# Model
- task: zero-shot retrieval baselines (no training)
- BM25: bm25s, k1=1.2, b=0.75; tokenizers: syllable, pyvi
- dense 1: BAAI/bge-m3 @ 5617a9f61b028005a4858fdac845db406aefb181 (pytorch_model.bin), raw text
- dense 2: bkai-foundation-models/vietnamese-bi-encoder @ 84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0 (model.safetensors), pyvi-segmented text
- cache: /root/.cache/huggingface (shared default), HF_HUB_OFFLINE=1 at run time

# Hyperparameters
- seed: 42
- encode batch size: 128; max_seq_len: 512 (bi-encoder capped by its own 256 limit); fp16
- retrieval depth k: 100; exact cosine search on GPU
- GPU policy: inference only, shared with the running ACL-CoT_Distillation job on GPU0 (approved by user 2026-11-23); CUDA_VISIBLE_DEVICES=0; no DDP, no training

# Outputs
- run dir: outputs/runs/baselines_v1_20261123/
  - metrics: outputs/runs/baselines_v1_20261123/metrics/*.json
  - log: outputs/runs/baselines_v1_20261123/run.log
  - PID: outputs/runs/baselines_v1_20261123/run.pid
  - exit code: outputs/runs/baselines_v1_20261123/exit_code
  - phase: outputs/runs/baselines_v1_20261123/phase
- report: run-outputs/out_baselines_v1.md

# History
- 2026-11-23: created.
