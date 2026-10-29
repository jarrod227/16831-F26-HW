"""
Make the figures for the HW3 report from the tensorboard logs in hw3/data.

    python rob831/scripts/plot_hw3.py

Writes figures/q1.png, figures/q2.png, figures/q3.png (any question whose
runs are missing is skipped).
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HW3_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')


def load_scalars(logdir):
    """Return {tag: (steps, values)} for every scalar in the run's event file."""
    event_files = glob.glob(os.path.join(logdir, 'events*'))
    if not event_files:
        return {}
    ea = EventAccumulator(event_files[0], size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))
    return out


def find_runs(data_dir, prefix):
    """Run folders are named <exp_name>_<env_name>_<timestamp>."""
    return sorted(glob.glob(os.path.join(data_dir, prefix + '_*')))


def mean_std_on_grid(curves, num_points=200):
    """Interpolate several (x, y) curves onto a shared x grid and average them."""
    x_lo = max(x[0] for x, _ in curves)
    x_hi = min(x[-1] for x, _ in curves)
    grid = np.linspace(x_lo, x_hi, num_points)
    ys = np.stack([np.interp(grid, x, y) for x, y in curves])
    return grid, ys.mean(axis=0), ys.std(axis=0)


def plot_q1(data_dir, out_dir):
    groups = [('q1_dqn', 'DQN', 'tab:blue'), ('q1_doubledqn', 'Double DQN', 'tab:red')]
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for prefix, label, color in groups:
        # 'q1_dqn_' so that 'q1_dqn' does not also match 'q1_doubledqn'
        runs = find_runs(data_dir, prefix + '_[0-9]')
        if not runs:
            print('Q1: no runs found for {}'.format(prefix))
            continue
        avg_curves, best_curves = [], []
        for run in runs:
            scalars = load_scalars(run)
            if 'Train_AverageReturn' not in scalars:
                continue
            avg_curves.append(scalars['Train_AverageReturn'])
            if 'Train_BestReturn' in scalars:
                best_curves.append(scalars['Train_BestReturn'])
        if not avg_curves:
            continue
        x, mean, std = mean_std_on_grid(avg_curves)
        ax.plot(x, mean, color=color, label='{} Train_AverageReturn ({} seeds)'.format(label, len(avg_curves)))
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        if best_curves:
            x, mean, _ = mean_std_on_grid(best_curves)
            ax.plot(x, mean, color=color, linestyle='--', label='{} Train_BestReturn'.format(label))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Return')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_title('LunarLander-v3: DQN vs Double DQN (mean $\\pm$ std over seeds)')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'q1.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', path)


def plot_eval(data_dir, out_dir, prefix, title, filename):
    runs = find_runs(data_dir, prefix)
    if not runs:
        print('{}: no runs found'.format(prefix))
        return
    scalars = load_scalars(runs[-1])  # most recent run with this exp_name
    if 'Eval_AverageReturn' not in scalars:
        print('{}: no Eval_AverageReturn logged'.format(prefix))
        return
    itr, mean = scalars['Eval_AverageReturn']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(itr, mean, color='tab:green', label='Eval_AverageReturn')
    if 'Eval_StdReturn' in scalars:
        _, std = scalars['Eval_StdReturn']
        ax.fill_between(itr, mean - std, mean + std, color='tab:green', alpha=0.2,
                        label='$\\pm$ Eval_StdReturn')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eval average return')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.join(HW3_DIR, 'data'))
    parser.add_argument('--out_dir', type=str, default=os.path.join(HW3_DIR, 'figures'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_q1(args.data_dir, args.out_dir)
    plot_eval(args.data_dir, args.out_dir, 'q2_10_10_CartPole-v0',
              'CartPole-v0: actor-critic (ntu=10, ngsptu=10)', 'q2.png')
    plot_eval(args.data_dir, args.out_dir, 'q3_10_10_InvertedPendulum-v4',
              'InvertedPendulum-v4: actor-critic (ntu=10, ngsptu=10)', 'q3.png')
