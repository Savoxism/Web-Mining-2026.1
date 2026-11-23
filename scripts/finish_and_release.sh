#!/usr/bin/env bash
# Finish the last experiment (second judge, Qwen3-32B), then release every GPU process of this project
# (vLLM Qwen3-4B / gpt-oss-120b / Qwen3-32B and the Streamlit demo) so another project can use the GPUs.
#   setsid nohup bash scripts/finish_and_release.sh > /dev/null 2>&1 &
# Progress: bash scripts/watch_final.sh
set -uo pipefail
cd "$(dirname "$0")/.."
S=outputs/runs/final_judge_20261124
mkdir -p "$S"
echo $$ > "$S/run.pid"
export HF_HUB_DISABLE_XET=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
stage() { echo "$1" > "$S/stage"; echo "[$(date -u +%FT%TZ)] $1" >> "$S/finish.log"; }

stage download
while pgrep -f "hf download Qwen/Qwen3-32B" > /dev/null; do sleep 20; done
# resume / verify (returns at once when every file is complete)
HF_HUB_OFFLINE=0 .venv/bin/hf download Qwen/Qwen3-32B --max-workers 16 >> "$S/download.log" 2>&1 || stage "download_failed"

if [[ "$(cat "$S/stage")" != "download_failed" ]]; then
  stage serve
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 setsid nohup .venv-vllm/bin/vllm serve Qwen/Qwen3-32B --served-model-name qwen3-32b \
    --host 127.0.0.1 --port 8300 --gpu-memory-utilization 0.85 --max-model-len 16384 > "$S/vllm_qwen32b.log" 2>&1 < /dev/null &
  echo $! > "$S/vllm_qwen32b.pid"
  for i in $(seq 1 180); do curl -s localhost:8300/v1/models > /dev/null && break; sleep 10; done
  if curl -s localhost:8300/v1/models > /dev/null; then
    stage rejudge
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/eval/rejudge_answers.py \
      --answers outputs/runs/notices_v2_20261123/eval_answers_final2/metrics/answers.json \
      --judge-url http://127.0.0.1:8300/v1 --out "$S/rejudge.json" > "$S/rejudge.log" 2>&1
    echo $? > "$S/rejudge_exit"
  else
    stage serve_failed
  fi
fi

stage release
# every process started from this project's virtual environments, by process group (vLLM engines included)
for pid in $(pgrep -f "/workspace/Web-Mining-2026.1/.venv(-vllm)?/bin/(python|vllm|streamlit)|\.venv-vllm/bin/vllm serve|\.venv/bin/streamlit run"); do
  [[ "$pid" == "$$" ]] && continue
  pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  [[ -n "$pg" && "$pg" != "$(ps -o pgid= -p $$ | tr -d ' ')" ]] && kill -TERM -- "-$pg" 2>/dev/null
done
sleep 30
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  if [[ "$(readlink -f /proc/$pid/cwd 2>/dev/null)" == /workspace/Web-Mining-2026.1* ]]; then kill -KILL "$pid" 2>/dev/null; fi
done
rm -f outputs/pids/vllm.pid outputs/pids/vllm_judge.pid outputs/pids/streamlit.pid
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$S/finish.log"
stage done
