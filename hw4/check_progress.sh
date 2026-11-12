#!/usr/bin/env bash
# Show, for each background run, the last iteration reached and its latest eval return.
cd "$(dirname "$0")"
echo "running python processes: $(pgrep -fc run_hw4_mb.py)"
for f in logs/*.log; do
    name=$(basename "$f" .log)
    itr=$(grep -o 'Iteration [0-9]*' "$f" | tail -1)
    ret=$(grep 'Eval_AverageReturn' "$f" | tail -1 | awk '{print $3}')
    err=$(grep -m1 -E 'Error|Traceback' "$f")
    printf '%-24s %-14s Eval=%-10s %s\n' "$name" "${itr:-starting}" "${ret:--}" "${err:+ERROR: $err}"
done
