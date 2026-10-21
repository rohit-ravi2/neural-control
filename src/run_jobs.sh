#!/bin/bash
# Run a jobs file through train_worm.py: run_jobs.sh <out_dir_name> <jobs_file> [workers]
# Each line of the jobs file is "<tag> <train_worm.py args...>". Outputs go to results/<out_dir_name>/.
cd "$(dirname "$0")"
OUT=$(cd .. && pwd)/results/$1; JOBS=$2; P=${3:-4}
mkdir -p "$OUT/logs"; [ "$(realpath "$JOBS")" = "$(realpath -m "$OUT/jobs.txt")" ] || cp "$JOBS" "$OUT/jobs.txt"
run() {
  name=$1; shift
  [ -f "$OUT/logs/$name.log" ] && return
  WORM_DEVICE=${WORM_DEVICE:-cuda} WORM_THREADS=2 WORM_OUT=$OUT python -u train_worm.py "$@" --tag "$name" > "$OUT/logs/$name.log" 2>&1
}
export -f run; export OUT
xargs -P "$P" -L 1 bash -c 'run "$@"' _ < "$OUT/jobs.txt"
