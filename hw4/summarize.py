"""
Print a compact text summary of every run in data/ so it can be pasted into chat.

    python summarize.py            # all runs
    python summarize.py q3         # only runs whose folder name contains "q3"
"""
import glob
import os
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


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
        tags = ['Train_AverageReturn', 'Eval_AverageReturn'] + [t for t in available if 'Loss' in t]
        for tag in tags:
            if tag in available:
                vals = [e.value for e in ea.Scalars(tag)]
                spec = '{:.3g}' if 'Loss' in tag else '{:.1f}'
                print('  {:<20} ({} itr): {}'.format(tag, len(vals), fmt(vals, spec)))


if __name__ == '__main__':
    main()
