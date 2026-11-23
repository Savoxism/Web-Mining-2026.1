#!/usr/bin/env bash
# Live view of the last experiments and the GPU release:  bash scripts/watch_final.sh   (Ctrl-C to quit)
cd "$(dirname "$0")/.."
S=outputs/runs/final_judge_20261124
G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; C=$'\e[36m'; B=$'\e[1m'; D=$'\e[2m'; N=$'\e[0m'
bar() {  # bar <done> <total> <colour>
  local d=$1 t=$2 c=$3 w=40 f p
  (( t <= 0 )) && t=1; (( d > t )) && d=$t
  f=$(( d * w / t )); p=$(( d * 100 / t ))
  printf "%s%s%s%s %3d%%" "$c" "$(printf '█%.0s' $(seq 1 $f) 2>/dev/null)" "$D$(printf '░%.0s' $(seq 1 $((w - f))) 2>/dev/null)" "$N" "$p"
}
order=(download serve rejudge release done)
while true; do
  clear
  st=$(cat "$S/stage" 2>/dev/null || echo "not started")
  printf "%sLexGraph-VN · last experiments and GPU release%s   %s (giờ VN)\n\n" "$B" "$N" "$(TZ=Asia/Ho_Chi_Minh date '+%H:%M:%S %d/%m')"
  for s in "${order[@]}"; do
    mark="$D○$N"; [[ "$st" == "$s" ]] && mark="$Y●$N"
    for x in "${order[@]}"; do [[ "$x" == "$st" ]] && break; [[ "$x" == "$s" ]] && mark="$G✔$N"; done
    [[ "$st" == "done" ]] && mark="$G✔$N"
    [[ "$st" == *failed* && "$s" == "${st%_failed}" ]] && mark="$R✘$N"
    printf "  %s %s\n" "$mark" "$s"
  done
  echo
  have=$(du -sbL /root/.cache/huggingface/hub/models--Qwen--Qwen3-32B/ 2>/dev/null | cut -f1); have=${have:-0}
  printf "  %-26s " "Qwen3-32B download"; bar $((have / 1000000)) 65530 "$C"; printf "  %s / 65.5 GB\n" "$(awk -v h="$have" "BEGIN{printf \"%.1f\", h/1e9}")"
  srv="${R}down${N}"; curl -s -m 2 localhost:8300/v1/models > /dev/null && srv="${G}ready${N}"
  printf "  %-26s %s\n" "Qwen3-32B server (8300)" "$srv"
  line=$(tr '\r' '\n' < "$S/rejudge.log" 2>/dev/null | grep -a "rejudge:" | tail -1)
  d=$(echo "$line" | grep -oE '[0-9]+/[0-9]+' | head -1)
  printf "  %-26s " "Second judge"; bar "${d%/*}" "${d#*/}" "$G"; printf "  %s\n" "${d:-waiting}"
  echo
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=', ' read -r i u t g; do
    printf "  GPU%s memory %-8s " "$i" ""; bar "$u" "$t" "$Y"; printf "  %6s / %s MiB, util %s%%\n" "$u" "$t" "$g"
  done
  echo
  [[ -f "$S/rejudge.json" ]] && printf "  %sResult:%s %s\n" "$B" "$N" "$(python3 -c "import json;d=json.load(open('$S/rejudge.json'));print('agreement', {k:v['kappa'] for k,v in d['agreement'].items()})" 2>/dev/null)"
  tail -3 "$S/finish.log" 2>/dev/null | sed "s/^/  $D/;s/$/$N/"
  [[ "$st" == "done" ]] && { printf "\n  %sDone: GPUs released.%s\n" "$G$B" "$N"; break; }
  sleep 5
done
