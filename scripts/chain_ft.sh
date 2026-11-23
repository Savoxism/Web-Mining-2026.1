#!/usr/bin/env bash
# Chain the remaining phases of run finetune_v1_20261123 so the GPUs never wait for a human:
#   train_bi done -> mine_ce (GPU0) + mine_ce_val (GPU1) -> train_ce (GPU0+1)
#   -> eval_ft (GPU0) + judge server gpt-oss-120b (GPU1) -> eval_answers
# Launch:  setsid nohup bash scripts/chain_ft.sh > outputs/logs/chain_ft.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
R=finetune_v1_20261123
D=outputs/runs/$R
BI=outputs/models/bge-m3-legal/final
CE_BASE=BAAI/bge-reranker-v2-m3; CE_REV=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
CE=outputs/models/reranker-legal/final

wait_ok() {  # wait for phases to finish; abort the chain if any failed
  for p in "$@"; do
    until [[ -f $D/$p/exit_code ]]; do sleep 20; done
    if [[ "$(cat $D/$p/exit_code)" != "0" ]]; then echo "$(date) phase $p failed, chain stopped"; exit 1; fi
    echo "$(date) phase $p ok"
  done
}

wait_ok train_bi
bash scripts/phase.sh $R mine_ce 0 .venv/bin/python scripts/train/mine_negatives.py --model $BI --split train \
  --n-neg 7 --max-queries 50000 --out data/train/ce_train.parquet
bash scripts/phase.sh $R mine_ce_val 1 .venv/bin/python scripts/train/mine_negatives.py --model $BI --split val \
  --n-neg 7 --max-queries 2000 --dump-candidates data/train/ce_val_candidates.parquet --out data/train/ce_val.parquet
wait_ok mine_ce mine_ce_val

bash scripts/phase.sh $R train_ce 0,1 .venv/bin/torchrun --nproc_per_node 2 --master_port 29512 \
  scripts/train/train_crossencoder.py --base $CE_BASE --revision $CE_REV --train data/train/ce_train.parquet \
  --val-candidates data/train/ce_val_candidates.parquet --out outputs/models/reranker-legal \
  --n-neg 7 --per-device-batch 64 --eval-steps 500 --epochs 1 --lr 1e-5
wait_ok train_ce

bash scripts/phase.sh $R eval_ft 0 .venv/bin/python scripts/eval/eval_pipeline.py --bi $BI --ce $CE \
  --out $D/eval_ft/metrics
# judge model on GPU1 (Qwen3-4B for the demo already runs there with a small memory share)
if ! curl -s localhost:8200/v1/models > /dev/null; then
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NCCL_NVLS_ENABLE=0 setsid nohup .venv-vllm/bin/vllm serve openai/gpt-oss-120b \
    --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a --served-model-name gpt-oss-120b --host 127.0.0.1 --port 8200 \
    --gpu-memory-utilization 0.70 --max-model-len 16384 > outputs/logs/vllm_judge.log 2>&1 < /dev/null &
  echo $! > outputs/pids/vllm_judge.pid
fi
wait_ok eval_ft
until curl -s localhost:8200/v1/models > /dev/null; do sleep 15; done
echo "$(date) judge ready"
bash scripts/phase.sh $R eval_answers 0 .venv/bin/python scripts/eval/eval_answers.py --bi $BI --ce $CE \
  --weights $D/eval_ft/metrics/results.json --out $D/eval_answers/metrics
wait_ok eval_answers
echo "$(date) chain done"
