#!/usr/bin/env bash
# Project-wide run watcher. Usage: INTERVAL=10 bash scripts/watch.sh
# One row per run phase under outputs/runs/<run_id>/<phase>/ (and legacy single-phase runs).
# Ctrl-C stops only this watcher.
INTERVAL="${INTERVAL:-10}"
cd "$(dirname "$0")/.." || exit 1
trap 'echo; exit 0' INT

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; D=$'\e[2m'; N=$'\e[0m'
else
  G=""; Y=""; R=""; D=""; N=""
fi
fmt() { local s=$1; printf '%02d:%02d:%02d' $(( s / 3600 )) $(( s % 3600 / 60 )) $(( s % 60 )); }
bar() { local pct=$1 w=20 f; f=$(( pct * w / 100 )); printf '%*s' "$f" '' | tr ' ' '#'; printf '%*s' $(( w - f )) '' | tr ' ' '-'; }

progress() {  # last tqdm / HF-trainer progress in a log: prints "pct step/total eta"
  local log=$1
  tail -c 4000 "$log" 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ \[[^]]*\]' | tail -1 |
    sed -E 's/^([0-9]+)%\|[^|]*\| *([0-9]+\/[0-9]+) \[([^<]*)<([^,]*).*/\1 \2 \4/'
}
metric() {  # last loss / eval metric in a log
  local log=$1 l
  l=$(grep -oE "'(loss|eval_val_cosine_mrr@10|eval_val_ndcg@10)': [0-9.]+" "$log" 2>/dev/null | tail -2 | tr '\n' ' ')
  echo "${l:-–}"
}

while true; do
  [[ -t 1 ]] && clear
  printf '%s  (refresh %ss)\n\n' "$(date '+%F %T')" "$INTERVAL"
  printf '%-24s %-18s %-8s %-22s %5s %-12s %-10s %-10s %s\n' RUN_ID PHASE STATUS PROGRESS PCT STEP ELAPSED ETA LATEST
  for d in outputs/runs/*/*/ outputs/runs/*/; do
    [[ -f "$d/start_time" || -f "$d/run.pid" ]] || continue
    [[ -f "$d/command" ]] || [[ -f "$d/phase" ]] || continue
    phase=$(basename "$d"); run=$(basename "$(dirname "$d")")
    [[ -f "$d/phase" ]] && { run=$phase; phase=$(awk '{print $2}' "$d/phase"); }
    start=$(cat "$d/start_time" 2>/dev/null || date +%s); now=$(date +%s); el=$(( now - start ))
    if [[ -f "$d/exit_code" ]]; then
      [[ "$(cat "$d/exit_code")" == "0" ]] && status=done || status=failed
      el=$(( $(stat -c %Y "$d/exit_code" 2>/dev/null || echo "$now") - start ))
    elif [[ -f "$d/run.pid" ]] && kill -0 "$(cat "$d/run.pid")" 2>/dev/null; then status=running
    else status=failed; fi
    read -r pct step eta <<< "$(progress "$d/run.log")"
    [[ $status == done ]] && { pct=100; eta="00:00"; }
    pct=${pct:-0}; step=${step:-–}; eta=${eta:-–}
    case $status in done) c=$G;; running) c=$Y;; failed) c=$R;; *) c=$D;; esac
    printf '%-24s %-18s %s%-8s%s [%s] %4s%% %-12s %-10s %-10s %s\n' "$run" "$phase" "$c" "$status" "$N" \
      "$(bar "$pct")" "$pct" "$step" "$(fmt $el)" "$eta" "$(metric "$d/run.log")"
  done
  sleep "$INTERVAL"
done
