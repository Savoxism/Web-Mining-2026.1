#!/usr/bin/env bash
# Project-wide run watcher. Usage: INTERVAL=10 bash scripts/watch.sh
# One row per run under outputs/runs/. Ctrl-C stops only this watcher.
INTERVAL="${INTERVAL:-10}"
cd "$(dirname "$0")/.." || exit 1
trap 'echo; exit 0' INT

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; D=$'\e[2m'; N=$'\e[0m'
else
  G=""; Y=""; R=""; D=""; N=""
fi

bar() { local pct=$1 w=20 f; f=$(( pct * w / 100 )); printf '%s%s' "$(printf '#%.0s' $(seq 1 $f) 2>/dev/null)" "$(printf -- '-%.0s' $(seq 1 $(( w - f ))) 2>/dev/null)"; }
fmt() { local s=$1; printf '%02d:%02d:%02d' $(( s / 3600 )) $(( s % 3600 / 60 )) $(( s % 60 )); }

while true; do
  [[ -t 1 ]] && clear
  printf '%s  (refresh %ss)\n\n' "$(date '+%F %T')" "$INTERVAL"
  printf '%-26s %-20s %-8s %-22s %5s %7s %-10s %-10s %s\n' RUN_ID PHASE STATUS PROGRESS PCT STEP ELAPSED ETA LATEST_METRIC
  for d in outputs/runs/*/; do
    [[ -d "$d" ]] || continue
    id=$(basename "$d")
    total=$(cat "$d/n_phases" 2>/dev/null || echo 1)
    read -r idx name status _ < "$d/phase" 2>/dev/null || { idx=0; name="-"; status=queued; }
    start=$(cat "$d/start_time" 2>/dev/null || date +%s)
    now=$(date +%s); el=$(( now - start ))
    if [[ -f "$d/exit_code" ]]; then
      [[ "$(cat "$d/exit_code")" == "0" ]] && status=done || status=failed
      el=$(( $(stat -c %Y "$d/exit_code" 2>/dev/null || echo "$now") - start ))
    elif [[ -f "$d/run.pid" ]] && ! kill -0 "$(cat "$d/run.pid")" 2>/dev/null; then
      status=failed
    fi
    finished=$(( idx - 1 )); [[ $status == done ]] && finished=$total
    pct=$(( finished * 100 / total ))
    eta="-"; [[ $finished -gt 0 && $status == running ]] && eta=$(fmt $(( el * (total - finished) / finished )))
    latest=$(ls -t "$d"/metrics/*.json 2>/dev/null | grep -v eda_stats | head -1)
    metric="-"
    [[ -n "$latest" ]] && metric="$(basename "$latest" .json) MRR@10=$(grep -o '"MRR@10": [0-9.]*' "$latest" | awk '{print $2}')"
    case $status in done) c=$G;; running) c=$Y;; failed) c=$R;; *) c=$D;; esac
    printf '%-26s %-20s %s%-8s%s [%s] %4s%% %7s %-10s %-10s %s\n' \
      "$id" "$name" "$c" "$status" "$N" "$(bar $pct)" "$pct" "$finished/$total" "$(fmt $el)" "$eta" "$metric"
  done
  sleep "$INTERVAL"
done
