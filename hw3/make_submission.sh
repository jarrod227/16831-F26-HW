#!/usr/bin/env bash
# Build submit.zip for Gradescope (HW3 Code) from inside the hw3 folder:
#     bash make_submission.sh
set -e
cd "$(dirname "$0")"
rm -rf submit submit.zip
mkdir -p submit/run_logs
# run logs: only the final experiment folders, without videos / checkpoints
for d in data/q1_dqn_[123]_* data/q1_doubledqn_[123]_* data/q2_10_10_* data/q3_10_10_*; do
    [ -d "$d" ] && cp -r "$d" submit/run_logs/
done
rm -rf submit/run_logs/*/gym submit/run_logs/*/*.pt
# code (same layout as the repo, no data/, no caches)
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude 'data' rob831/ submit/rob831/
cat > submit/README.md <<'MD'
# HW3 submission

Code is in `rob831/` (same layout as the starter repo); logs are in `run_logs/`.

Reproduce the figures (from the `hw3` folder, with `run_logs/` copied to `data/`):

    python rob831/scripts/plot_hw3.py        # writes figures/q1.png, q2.png, q3.png

Experiments were run with the exact commands from the assignment PDF
(`run_hw3_dqn.py` for q1_*, `run_hw3_actor_critic.py` for q2_10_10 / q3_10_10),
all with `--no_gpu` and default hyperparameters.
MD
zip -qr submit.zip submit -x "*.DS_Store"
du -sh submit.zip
echo "submit.zip ready (must be < 15MB)"
