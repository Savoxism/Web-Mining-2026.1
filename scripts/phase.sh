#!/usr/bin/env bash
# Run one phase in the background with its own log / exit code:
#   bash scripts/phase.sh <run_id> <phase> <gpus> <command...>
# e.g. bash scripts/phase.sh finetune_v1_20261123 temporal 0 .venv/bin/python scripts/kg/build_temporal.py ...
set -uo pipefail
RUN_ID="$1"; PHASE="$2"; GPUS="$3"; shift 3
DIR="outputs/runs/${RUN_ID}/${PHASE}"
mkdir -p "$DIR"
rm -f "$DIR/exit_code"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_NVLS_ENABLE=0  # NVLS multicast fails under the CUDA MPS daemon on this server
export CUDA_VISIBLE_DEVICES="$GPUS"
date +%s > "$DIR/start_time"
echo "$*" > "$DIR/command"
setsid nohup bash -c "\"\$@\" > '$DIR/run.log' 2>&1; echo \$? > '$DIR/exit_code'" _ "$@" > /dev/null 2>&1 < /dev/null &
echo $! > "$DIR/run.pid"
echo "launched $PHASE on GPU(s) $GPUS (pid $(cat $DIR/run.pid))"
