#!/usr/bin/env bash
# Sweep vllm bench serve over IN x OU grid and archive fused_marlin_moe.log each run.
#
# Usage:
#   ./run_bench_in_ou_sweep.sh /path/to/your_bench.sh [args for bench script...]
#
# The wrapped script must read input/output lengths from positional args 4 and 5:
#   IN=${4:-100}
#   OU=${5:-1024}
#
# Example (NC/NP as $2/$3, model path etc. set inside bench script):
#   ./run_bench_in_ou_sweep.sh ./bench_serve.sh 8 100
#
# Environment:
#   MOE_LOG_SRC   source log path (default: /tmp/fused_marlin_moe.log)
#   MOE_LOG_DIR   where to mv archived logs (default: ./moe_logs)
#   IN_LIST       space-separated IN values (override default grid)
#   OU_LIST       space-separated OU values (override default grid)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bench_script.sh> [args passed to bench script...]" >&2
  exit 1
fi

BENCH_SCRIPT=$1
shift

if [[ ! -x "$BENCH_SCRIPT" && ! -f "$BENCH_SCRIPT" ]]; then
  echo "error: bench script not found: $BENCH_SCRIPT" >&2
  exit 1
fi

MOE_LOG_SRC=${MOE_LOG_SRC:-/tmp/fused_marlin_moe.log}
MOE_LOG_DIR=${MOE_LOG_DIR:-./moe_logs}
mkdir -p "$MOE_LOG_DIR"

IN_VALUES=(${IN_LIST:-100 200 500 600 1000})
OU_VALUES=(${OU_LIST:-512 1024 2048 4096 8192})

run_one() {
  local in_len=$1
  local ou_len=$2
  local archived="${MOE_LOG_DIR}/${in_len}_${ou_len}_eager.log"

  echo "============================================================"
  echo "run IN=${in_len} OU=${ou_len}"
  echo "bench: ${BENCH_SCRIPT} $* ${in_len} ${ou_len}"
  echo "archive log -> ${archived}"
  echo "============================================================"

  # Drop stale log so we only capture this run.
  rm -f "$MOE_LOG_SRC"

  # Positional args 4/5 are IN/OU in the wrapped script.
  bash "$BENCH_SCRIPT" "$@" "$in_len" "$ou_len"

  if [[ -f "$MOE_LOG_SRC" ]]; then
    sleep 1
    mv -f "$MOE_LOG_SRC" "$archived"
    echo "saved ${archived}"
  else
    echo "warning: ${MOE_LOG_SRC} not found; skip archive for IN=${in_len} OU=${ou_len}" >&2
  fi
}

for in_len in "${IN_VALUES[@]}"; do
  for ou_len in "${OU_VALUES[@]}"; do
    run_one "$in_len" "$ou_len" "$@"
  done
done

echo "done: ${#IN_VALUES[@]} x ${#OU_VALUES[@]} runs, logs in ${MOE_LOG_DIR}/"
