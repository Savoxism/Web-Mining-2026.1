#!/usr/bin/env bash
# Start Qwen3-4B (vLLM, port 8100) and the Streamlit demo (port 8501) on the GPU server.
#   bash scripts/serve_demo.sh            # start both
#   bash scripts/serve_demo.sh stop       # stop both
# From your laptop:  ssh -N -L 8501:localhost:8501 H200_ANNP36   then open http://localhost:8501
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/pids
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
GPU="${DEMO_GPU:-0}"

if [[ "${1:-start}" == "stop" ]]; then
  for p in outputs/pids/vllm.pid outputs/pids/streamlit.pid; do
    [[ -f $p ]] && kill "$(cat $p)" 2>/dev/null && echo "stopped $(basename $p .pid)"; rm -f $p
  done
  exit 0
fi

if ! curl -s localhost:8100/v1/models >/dev/null; then
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup .venv-vllm/bin/vllm serve Qwen/Qwen3-4B \
    --revision 1cfa9a7208912126459214e8b04321603b3df60c --served-model-name qwen3-4b \
    --host 127.0.0.1 --port 8100 --gpu-memory-utilization "${VLLM_MEM:-0.12}" --max-model-len 16384 \
    > outputs/logs/vllm.log 2>&1 < /dev/null &
  echo $! > outputs/pids/vllm.pid
  echo "vLLM starting (log: outputs/logs/vllm.log)"
  for i in $(seq 1 90); do curl -s localhost:8100/v1/models >/dev/null && break; sleep 5; done
  curl -s localhost:8100/v1/models >/dev/null && echo "vLLM ready" || echo "vLLM not ready yet, check the log"
fi

CUDA_VISIBLE_DEVICES=$GPU setsid nohup .venv/bin/streamlit run demo/app.py --server.port 8501 \
  --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false \
  > outputs/logs/streamlit.log 2>&1 < /dev/null &
echo $! > outputs/pids/streamlit.pid
echo "Streamlit on 127.0.0.1:8501 (log: outputs/logs/streamlit.log)"
