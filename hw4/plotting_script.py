"""
Make every figure in the HW4 report from the logged experiments.

    python plotting_script.py                       # reads rob831/data
    python plotting_script.py --data_dir run_logs   # reads the submitted logs

All returns are read from the tensorboard event files; nothing is hard-coded.
Q1 figures are the itr_0_predictions.png images the training code saves, copied
over unchanged. Everything is written to figures/.
"""
import argparse
import glob
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HW4_DIR = os.path.dirname(os.path.realpath(__file__))


def find_run(data_dir, exp_name):
    """Most recent run folder for an exp_name (folders are hw4_<exp_name>_<env>_<time>)."""
    runs = sorted(glob.glob(os.path.join(data_dir, 'hw4_{}_*-hw4_part1-v0_*'.format(exp_name))))
    if not runs:
        print('missing run: {}'.format(exp_name))
        return None
    return runs[-1]


def load_scalar(run, tag):
    """(iterations, values) for one scalar tag of a run's event file."""
    event_files = glob.glob(os.path.join(run, 'events*'))
    if not event_files:
        return [], []
    ea = EventAccumulator(event_files[0], size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def save(fig, out_dir, name):
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', path)


def plot_q1(data_dir, out_dir):
    for exp in ['q1_cheetah_n500_arch1x32', 'q1_cheetah_n5_arch2x250', 'q1_cheetah_n500_arch2x250']:
        run = find_run(data_dir, exp)
        if run is None:
            continue
        src = os.path.join(run, 'itr_0_predictions.png')
        dst = os.path.join(out_dir, exp + '.png')
        shutil.copyfile(src, dst)
        print('wrote', dst)


def plot_q2(data_dir, out_dir):
    run = find_run(data_dir, 'q2_obstacles_singleiteration')
    if run is None:
        return
    _, train = load_scalar(run, 'Train_AverageReturn')
    _, evals = load_scalar(run, 'Eval_AverageReturn')
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([0], train, s=80, label='Train_AverageReturn (random actions)')
    ax.scatter([0], evals, s=80, marker='s', label='Eval_AverageReturn (MPC)')
    ax.set_xticks([0])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average return')
    ax.set_title('Q2: obstacles, single iteration')
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, out_dir, 'q2_obstacles.png')


def plot_q3(data_dir, out_dir):
    for exp, env in [('q3_obstacles', 'Obstacles'), ('q3_reacher', 'Reacher'), ('q3_cheetah', 'Cheetah')]:
        run = find_run(data_dir, exp)
        if run is None:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for tag, marker in [('Train_AverageReturn', 'o'), ('Eval_AverageReturn', 's')]:
            itr, val = load_scalar(run, tag)
            ax.plot(itr, val, marker=marker, label=tag)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average return')
        ax.set_title('Q3: {} (MBRL with on-policy data)'.format(env))
        ax.grid(alpha=0.3)
        ax.legend()
        save(fig, out_dir, exp + '.png')


def plot_q4(data_dir, out_dir):
    # The default configuration (H = 10, K = 1000, N = 3) is exactly the
    # q4_reacher_numseq1000 run, so it doubles as the H = 10 reference curve.
    groups = [
        ('horizon', 'Planning horizon', [('q4_reacher_horizon5', 'H = 5'),
                                         ('q4_reacher_numseq1000', 'H = 10 (default; numseq1000 run)'),
                                         ('q4_reacher_horizon15', 'H = 15'),
                                         ('q4_reacher_horizon30', 'H = 30')]),
        ('numseq', 'Number of candidate action sequences', [('q4_reacher_numseq100', 'K = 100'),
                                                           ('q4_reacher_numseq1000', 'K = 1000')]),
        ('ensemble', 'Ensemble size', [('q4_reacher_ensemble1', 'N = 1'),
                                       ('q4_reacher_ensemble3', 'N = 3'),
                                       ('q4_reacher_ensemble5', 'N = 5')]),
    ]
    for key, title, runs in groups:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for exp, label in runs:
            run = find_run(data_dir, exp)
            if run is None:
                continue
            itr, val = load_scalar(run, 'Eval_AverageReturn')
            ax.plot(itr, val, marker='o', label=label)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Eval_AverageReturn')
        ax.set_title('Q4: Reacher, effect of {}'.format(title.lower()))
        ax.grid(alpha=0.3)
        ax.legend()
        save(fig, out_dir, 'q4_reacher_{}.png'.format(key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.join(HW4_DIR, 'rob831', 'data'))
    parser.add_argument('--out_dir', default=os.path.join(HW4_DIR, 'figures'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_q1(args.data_dir, args.out_dir)
    plot_q2(args.data_dir, args.out_dir)
    plot_q3(args.data_dir, args.out_dir)
    plot_q4(args.data_dir, args.out_dir)
