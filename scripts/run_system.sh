#!/usr/bin/env bash
# Build the system with base models: temporal layer (version alignment + extended corpus) and evaluation.
#   setsid nohup bash scripts/run_system.sh system_base_v1 > /dev/null 2>&1 < /dev/null &
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
BI=BAAI/bge-m3; BI_REV=5617a9f61b028005a4858fdac845db406aefb181
CE=BAAI/bge-reranker-v2-m3; CE_REV=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
PHASES=(temporal eval_base)
echo "${#PHASES[@]}" > "${RUN_DIR}/n_phases"

run_phase() {
  local i="$1" name="$2"; shift 2
  echo "$i ${name} running $(date +%s)" > "${RUN_DIR}/phase"
  echo "===== [$(date -u +%FT%TZ)] phase ${i}/${#PHASES[@]}: ${name} =====" >> "${LOG}"
  "$@" >> "${LOG}" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "$i ${name} failed $(date +%s)" > "${RUN_DIR}/phase"
    echo "${rc}" > "${RUN_DIR}/exit_code"; exit "${rc}"
  fi
}

START=$(date +%s); echo "${START}" > "${RUN_DIR}/start_time"
run_phase 1 temporal $PY scripts/kg/build_temporal.py --model $BI --revision $BI_REV \
  --corpus data/processed_v2/corpus.parquet --out data/kg/temporal
cp data/kg/temporal/temporal_stats.json "${RUN_DIR}/metrics/"
run_phase 2 eval_base $PY scripts/eval/eval_pipeline.py --bi $BI --bi-revision $BI_REV \
  --ce $CE --ce-revision $CE_REV --out "${RUN_DIR}/metrics/eval"

echo "${#PHASES[@]} done done $(date +%s)" > "${RUN_DIR}/phase"
echo 0 > "${RUN_DIR}/exit_code"
