#!/usr/bin/env bash
# Q3 (3 runs) + Q4 (8 runs), all launched in the background at once.
# They keep running after you log out. Output of each run goes to logs/<name>.log.
# Usage (from the hw4 folder, rob831 env active):  bash run_q3_q4.sh
cd "$(dirname "$0")"
mkdir -p logs
S=rob831/hw4_part1/scripts/run_hw4_mb.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {  # launch <log name> <args...>
    name=$1; shift
    nohup python -u $S "$@" > logs/$name.log 2>&1 &
    echo "started $name (pid $!)"
}

launch q3_obstacles --exp_name q3_obstacles --env_name obstacles-hw4_part1-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --n_iter 12 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
launch q3_reacher --exp_name q3_reacher --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 5000 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
launch q3_cheetah --exp_name q3_cheetah --env_name cheetah-hw4_part1-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 20 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

launch q4_reacher_horizon5 --exp_name q4_reacher_horizon5 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 5 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
launch q4_reacher_horizon15 --exp_name q4_reacher_horizon15 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
launch q4_reacher_horizon30 --exp_name q4_reacher_horizon30 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 30 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
launch q4_reacher_numseq100 --exp_name q4_reacher_numseq100 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 100 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
launch q4_reacher_numseq1000 --exp_name q4_reacher_numseq1000 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_num_action_sequences 1000 --mpc_action_sampling_strategy 'random'
launch q4_reacher_ensemble1 --exp_name q4_reacher_ensemble1 --env_name reacher-hw4_part1-v0 --ensemble_size 1 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
launch q4_reacher_ensemble3 --exp_name q4_reacher_ensemble3 --env_name reacher-hw4_part1-v0 --ensemble_size 3 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
launch q4_reacher_ensemble5 --exp_name q4_reacher_ensemble5 --env_name reacher-hw4_part1-v0 --ensemble_size 5 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

echo
echo "All 11 runs started. Check progress with:  bash check_progress.sh"
