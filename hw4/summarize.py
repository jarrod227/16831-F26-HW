"""
Print a compact text summary of every run in data/ so it can be pasted into chat.

    python summarize.py            # all runs
    python summarize.py q3         # only runs whose folder name contains "q3"
"""
import glob
import os
import sys
import time

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rob831', 'data')
N_ITER = {'q2_': 1, 'q3_obstacles': 12, 'q3_reacher': 15, 'q3_cheetah': 20, 'q4_': 15}


def total_iters(name):
    for key, n in N_ITER.items():
        if key in name:
            return n
    return None


def fmt(values, spec):
    return ' '.join(spec.format(v) for v in values)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ''
    for run in sorted(glob.glob(os.path.join(DATA, 'hw4_*'))):
        name = os.path.basename(run)
        if pattern not in name:
            continue
        print('=== ' + name)

        losses = os.path.join(run, 'itr_0_losses.npy')
        if os.path.exists(losses):
            l = np.load(losses)
            print('  model loss: first {:.3f}  last {:.3f}  ({} steps)'.format(l[0], l[-1], len(l)))
        mpe_files = sorted(glob.glob(os.path.join(run, 'itr_*_mpe.txt')))
        if mpe_files:
            print('  MPE (last itr): {:.4f}'.format(float(np.loadtxt(mpe_files[-1]))))

        events = glob.glob(os.path.join(run, 'events*'))
        if not events:
            continue
        ea = EventAccumulator(events[0], size_guidance={'scalars': 0})
        ea.Reload()
        available = ea.Tags()['scalars']
        if 'Eval_AverageReturn' in available:
            walls = [e.wall_time for e in ea.Scalars('Eval_AverageReturn')]
            n_total = total_iters(name)
            line = '  progress: {}/{} itr'.format(len(walls), n_total)
            if len(walls) >= 2:
                per_itr = (walls[-1] - walls[0]) / (len(walls) - 1)
                last_itr = walls[-1] - walls[-2]
                line += ', avg {:.1f} / last {:.1f} min/itr'.format(per_itr / 60, last_itr / 60)
                if n_total and len(walls) < n_total:
                    # the most recent iteration reflects the current GPU load best
                    left = (n_total - len(walls)) * last_itr - (time.time() - walls[-1])
                    line += ', ~{:.0f} min left'.format(max(left, 0) / 60)
                elif n_total and len(walls) >= n_total:
                    line += ', DONE'
            line += ', last update {:.0f} min ago'.format((time.time() - walls[-1]) / 60)
            print(line)
        tags = ['Train_AverageReturn', 'Eval_AverageReturn'] + [t for t in available if 'Loss' in t]
        for tag in tags:
            if tag in available:
                vals = [e.value for e in ea.Scalars(tag)]
                spec = '{:.3g}' if 'Loss' in tag else '{:.1f}'
                print('  {:<20} ({} itr): {}'.format(tag, len(vals), fmt(vals, spec)))


if __name__ == '__main__':
    main()
