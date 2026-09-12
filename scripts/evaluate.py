"""Offline rollout eval at a sample size that can actually separate checkpoints.

In-run eval runs 3 rollouts per task. At p=0.35 that is a 95% interval of about
+/-31 points -- wider than any effect we are trying to measure, which is why
run 9's eval/mean swung 0-56% all night without the policy necessarily changing.

Same rollout code path as Agent.eval (agent.test, unchanged), just more samples
and a Wilson interval on the result.

    python scripts/evaluate.py checkpoints/run9/vla_network.e*
    python scripts/evaluate.py --rollouts 30 --tasks all ckpt

Set UNFREEZE_LAST_N_LAYERS only if you want the backbone trainable; loading an
unfrozen checkpoint works either way, the names are matched with strict=False.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, EVAL_TASKS
from tasks import TASKS


def wilson(successes, n, z=1.96):
    """Score interval. Normal approximation is useless near 0 and 1, and both
    ends are live here -- microwave sits at 0, hinge_cabinet has read 1.0."""
    if not n:
        return 0.0, 1.0
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(agent, checkpoint, tasks, rollouts):
    agent.model.load_checkpoint(checkpoint)

    result = {"checkpoint": checkpoint, "rollouts": rollouts, "tasks": {}}
    for task in tasks:
        start = time.time()
        successes = sum(agent.test(task) for _ in range(rollouts))
        low, high = wilson(successes, rollouts)
        result["tasks"][task] = {"successes": successes, "rate": successes / rollouts,
                                 "ci": [low, high]}
        print(f"  {task:<16} {successes:>3}/{rollouts}  {successes / rollouts:>6.1%}"
              f"  [{low:.1%}, {high:.1%}]  {time.time() - start:.0f}s", flush=True)

    total = sum(t["successes"] for t in result["tasks"].values())
    n = rollouts * len(tasks)
    low, high = wilson(total, n)
    # Equal n per task, so the pooled rate is exactly eval/mean -- but with an
    # interval attached, which is the whole point of running this.
    result["mean"] = {"successes": total, "n": n, "rate": total / n, "ci": [low, high]}
    print(f"  {'MEAN':<16} {total:>3}/{n}  {total / n:>6.1%}  [{low:.1%}, {high:.1%}]",
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--tasks", nargs="+", default=EVAL_TASKS,
                        help="task names, or 'all' for every task in TASKS")
    parser.add_argument("--out", default=None, help="write results as json")
    args = parser.parse_args()

    tasks = list(TASKS) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        sys.exit(f"unknown tasks: {unknown}\n\ntasks:\n  " + "\n  ".join(TASKS))

    missing = [c for c in args.checkpoints if not os.path.isfile(c)]
    if missing:
        sys.exit(f"no such checkpoint: {missing}")

    # One agent for every checkpoint -- the VLM load is the slow part and the
    # weights we swap are only the head plus any unfrozen layers.
    agent = Agent(eval=True)

    results = []
    for checkpoint in args.checkpoints:
        print(f"\n{checkpoint}", flush=True)
        results.append(evaluate(agent, checkpoint, tasks, args.rollouts))

    print("\n" + "=" * 72)
    print(f"{'checkpoint':<34}{'mean':>8}{'95% CI':>22}")
    for r in sorted(results, key=lambda r: -r["mean"]["rate"]):
        m = r["mean"]
        ci = "[{:.1%}, {:.1%}]".format(*m["ci"])
        print(f"{os.path.basename(r['checkpoint']):<34}{m['rate']:>8.1%}{ci:>22}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
