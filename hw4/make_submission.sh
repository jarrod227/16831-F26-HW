#!/usr/bin/env bash
# Build submit.zip for Gradescope (HW4 Code). Run from the hw4 folder:
#     bash make_submission.sh
set -e
cd "$(dirname "$0")"
rm -rf submit submit.zip
mkdir -p submit/run_logs submit/rob831

# run logs: the latest run of every exp_name, without videos / checkpoints
for exp in q1_cheetah_n500_arch1x32 q1_cheetah_n5_arch2x250 q1_cheetah_n500_arch2x250 \
           q2_obstacles_singleiteration q3_obstacles q3_reacher q3_cheetah \
           q4_reacher_horizon5 q4_reacher_horizon15 q4_reacher_horizon30 \
           q4_reacher_numseq100 q4_reacher_numseq1000 \
           q4_reacher_ensemble1 q4_reacher_ensemble3 q4_reacher_ensemble5; do
    run=$(ls -d rob831/data/hw4_${exp}_*-hw4_part1-v0_* 2>/dev/null | sort | tail -1)
    if [ -z "$run" ]; then echo "MISSING: $exp"; continue; fi
    cp -r "$run" submit/run_logs/
done
find submit/run_logs \( -name '*.pt' -o -name '*.mp4' -o -name '*.gif' \) -delete

# code, same layout as the repo, without data/ or caches
rsync -a --exclude '__pycache__' --exclude '*.pyc' rob831/hw4_part1 submit/rob831/
cp plotting_script.py submit/

cat > submit/README.md <<'MD'
# HW4 submission

- `rob831/hw4_part1/`: code (same layout as the starter repo).
- `run_logs/`: one folder per experiment, exactly as written by `run_hw4_mb.py`.
- `plotting_script.py`: regenerates every figure in the report from `run_logs/`
  (returns are read from the tensorboard event files; Q1 figures are the
  `itr_0_predictions.png` files the training code saves):

      python plotting_script.py --data_dir run_logs --out_dir figures

All experiments use the exact commands from the assignment PDF.
MD

zip -qr submit.zip submit -x "*.DS_Store"
echo "run logs: $(ls submit/run_logs | wc -l) / 15"
du -sh submit.zip
echo "submit.zip ready (must be < 15MB)"
