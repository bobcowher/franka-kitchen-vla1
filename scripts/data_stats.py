"""Print per-task episode and step counts for the collected demos.

Reads only the small arrays out of each shard; camera_scene is never inflated,
so this runs in seconds over a 19 GB dataset.
"""
import glob
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

path = sys.argv[1] if len(sys.argv) > 1 else 'dataset'

files = sorted(glob.glob(os.path.join(path, '**', '*.npz'), recursive=True))
if not files:
    print(f"No npz files found under {path}")
    sys.exit(1)

tasks = {}
for filename in files:
    shard = np.load(filename)
    lengths, successes = tasks.setdefault(str(shard['task_name']), ([], 0))
    lengths.append(len(shard['action']))
    if shard['reward'].sum() > 0:
        tasks[str(shard['task_name'])] = (lengths, successes + 1)

print(f"{'task':<16}{'eps':>6}{'steps':>9}{'min':>6}{'mean':>7}{'max':>6}{'success':>9}")
for task in sorted(tasks):
    lengths, successes = tasks[task]
    print(f"{task:<16}{len(lengths):>6}{sum(lengths):>9}{min(lengths):>6}"
          f"{np.mean(lengths):>7.0f}{max(lengths):>6}"
          f"{successes:>6}/{len(lengths)}")

total_steps = sum(sum(lengths) for lengths, _ in tasks.values())
print(f"\n{len(files)} shards, {total_steps} steps, "
      f"{total_steps * 896 * 896 * 3 / 1e9:.1f} GB of frames at 896")
