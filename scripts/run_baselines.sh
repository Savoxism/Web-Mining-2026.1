#!/usr/bin/env bash
# Sequential baseline pipeline. Launch with:
#   nohup bash scripts/run_baselines.sh baselines_v1_20261123 > /dev/null 2>&1 &
set -uo pipefail

RUN_ID="${1:?run_id required}"
RUN_DIR="outputs/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/metrics"
LOG="${RUN_DIR}/run.log"
echo $$ > "${RUN_DIR}/run.pid"
rm -f "${RUN_DIR}/exit_code"

export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=".venv/bin/python"
OUT="${RUN_DIR}/metrics"

PHASES=(prepare bm25_syllable bm25_pyvi dense_bge_m3 dense_vn_bi_encoder)
echo "${#PHASES[@]}" > "${RUN_DIR}/n_phases"

run_phase() {
  local i="$1" name="$2"; shift 2
  echo "$i ${name} running $(date +%s)" > "${RUN_DIR}/phase"
  echo "===== [$(date -u +%FT%TZ)] phase ${i}/${#PHASES[@]}: ${name} =====" >> "${LOG}"
  "$@" >> "${LOG}" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "$i ${name} failed $(date +%s)" > "${RUN_DIR}/phase"
    echo "===== phase ${name} FAILED rc=${rc} =====" >> "${LOG}"
    echo "${rc}" > "${RUN_DIR}/exit_code"
    exit "${rc}"
  fi
}

START=$(date +%s); echo "${START}" > "${RUN_DIR}/start_time"
run_phase 1 prepare $PY scripts/prepare_data.py \
  --input data/soict-hackathon-2025-legal-corpus/train-00000-of-00001.parquet --out data/processed --seed 42
cp data/processed/eda_stats.json "${OUT}/"
run_phase 2 bm25_syllable $PY scripts/bm25_baseline.py --data data/processed --split test --tokenizer syllable --out "${OUT}"
run_phase 3 bm25_pyvi $PY scripts/bm25_baseline.py --data data/processed --split test --tokenizer pyvi --out "${OUT}"
run_phase 4 dense_bge_m3 $PY scripts/dense_baseline.py --model BAAI/bge-m3 --split test --out "${OUT}"
run_phase 5 dense_vn_bi_encoder $PY scripts/dense_baseline.py \
  --model bkai-foundation-models/vietnamese-bi-encoder --segment pyvi --split test --out "${OUT}"

echo "${#PHASES[@]} done done $(date +%s)" > "${RUN_DIR}/phase"
echo "===== [$(date -u +%FT%TZ)] all phases done in $(( $(date +%s) - START ))s =====" >> "${LOG}"
echo 0 > "${RUN_DIR}/exit_code"
