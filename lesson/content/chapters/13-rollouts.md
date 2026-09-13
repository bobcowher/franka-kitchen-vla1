---
title: "Rollouts, and how the eval lied"
part: "Part III · Knowing whether it works"
chapter: 13
weight: 13
standfirst: "The architecture worked long before we could tell. Three separate failures, all of them in the measurement."
---

Training loss is not the objective. The objective is whether the arm opens the
cabinet. So the training loop stops every 2,500 epochs and runs rollouts:
three tasks, three attempts each.

<p class="filename">Filename: <strong>agent.py</strong></p>

```python
EVAL_TASKS = ["microwave", "hinge cabinet", "top burner"]
EVAL_ROLLOUTS = 3


def eval(self, epoch, summary_writer):
    rates = []
    for task in EVAL_TASKS:
        rate = sum(self.test(task) for _ in range(EVAL_ROLLOUTS)) / EVAL_ROLLOUTS
        summary_writer.add_scalar(f"eval/{task.replace(' ', '_')}", rate, epoch)
        rates.append(rate)
    # Per-task rates are 0, 0.33, 0.67 or 1, so the mean is the signal.
    summary_writer.add_scalar("eval/mean", sum(rates) / len(rates), epoch)
    # Otherwise the checkpoint on disk is whichever eval ran last.
    torch.save(self.model.head.state_dict(), f"{self.model.checkpoint_file}.e{epoch}")
```

Three rollouts per task was a deliberate, correct decision at the start.
Rollouts are slow, early iteration matters more than precision, and any success
above zero was the win condition. It was right on day one.

It stopped being right, and nothing announced the transition.

## The arithmetic nobody ran

`eval/mean` is the average of three independent `Binomial(3, p)/3` estimates.
Its standard deviation is `√(p(1−p)/9)`.

At a true success rate of `p = 0.35`, that is **0.159** — a 95% interval of
roughly **±31 percentage points**.

<div class="finding">
<span class="note-label">The consequence</span>
An overnight run produced about 40 evaluation points swinging between 0% and
56%, and we read that curve for hours looking for trends: a peak at 40K, a
collapse after it, a recovery. <strong>The whole pattern is what a single
unchanging policy looks like sampled forty times at that resolution.</strong>
There was never a trend to find.
</div>

## Measuring it properly

No new ideas required, just more samples. Load any checkpoint, run it through
the *same* `agent.test()` path at arbitrary `n`, and put an interval on the
result.

<p class="filename">Filename: <strong>scripts/evaluate.py</strong></p>

```python
def wilson(successes, n, z=1.96):
    """Score interval. The normal approximation is useless near 0 and 1, and
    both ends are live here -- microwave sits at 0, hinge_cabinet has read 1."""
    if not n:
        return 0.0, 1.0
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(agent, checkpoint, tasks, rollouts):
    agent.model.load_checkpoint(checkpoint)
    for task in tasks:
        successes = sum(agent.test(task) for _ in range(rollouts))
        low, high = wilson(successes, rollouts)
        print(f"  {task:<16} {successes:>3}/{rollouts}  "
              f"{successes / rollouts:>6.1%}  [{low:.1%}, {high:.1%}]")
```

A <em class="term">Wilson interval</em> is the right one here because the
ordinary normal approximation misbehaves near 0 and 1, and both ends are live:
one task sits at zero, another has read 100%.

Rollouts parallelize almost perfectly — each spends most of its wall clock
waiting on the physics simulator, not the GPU — so run seven checkpoints at
once on one card.

1,050 rollouts later, at 50 per task instead of 3:

| checkpoint | in-run (n=9) | offline (n=100) | 95% CI |
|---|---|---|---|
| e40000 | 56% | **61.0%** | [51.2, 70.0] |
| e45000 | 44% | 60.0% | [50.2, 69.1] |
| e42500 | 33% | 52.0% | [42.3, 61.5] |
| e97500 | 44% | 47.0% | [37.5, 56.7] |
| e70000 | 44% | 35.0% | [26.4, 44.7] |
| e27500 | 22% | 32.0% | [23.7, 41.7] |
| e100000 | 0% | **0.0%** | [0.0, 3.7] |

## Failure one: the default artifact was a brick

`e100000` is the final checkpoint, what you get if you train eleven hours and
keep the weights at the end.

**Zero successes in 100 rollouts.** The interval tops out at 3.7%, so this is
not an unlucky sample. The policy collapsed somewhere in the last 2,500 epochs.

Meanwhile `train/loss` spent that window in its best-ever region, 0.04 to 0.09,
declining smoothly. The loss curve reported nothing.

<div class="trap">
<span class="note-label">Trap</span>
In behavior cloning, training loss and task success are loosely coupled and can
decouple completely. Any pipeline that keeps final weights is one bad window
away from shipping a policy that does nothing, with a training curve that looks
perfect. Keep the best evaluated checkpoint, and evaluate often enough that
"best" means something. This is why Chapter 9 snapshots the head at every
evaluation: at 85 KB each you can afford forty of them.
</div>

## Failure two: the ranking was wrong

`e45000` and `e70000` both read 44% in-run. They are 60% and 35%, with
non-overlapping intervals. Nine rollouts could not see a real 25-point gap.

Every decision made against that ranking overnight was made against noise.

The per-task columns hold something subtler. The checkpoints are not getting
uniformly better or worse. They are **trading tasks against each other**:

| checkpoint | hinge cabinet | top burner |
|---|---|---|
| e45000 | 86% | 34% |
| e40000 | 68% | 54% |
| e97500 | 26% | 68% |

The mean stays roughly flat while the policy drifts sideways, swapping which
task it is good at. Call that interference: 21,129 parameters asked to hold
several behaviors over a fixed representation. It is a far better argument for
unfreezing than any trend in `eval/mean`.

## Failure three: we never looked at the best task

The evaluation set was three tasks. The model trains on **seven**. Four had
never been rolled out, not once, in any run.

The question that unravelled this was about wasted compute: *why are we testing
microwave when we know it never succeeds?* The answer turned out to be worse
than waste.

| task | rate | 95% CI |
|---|---|---|
| **slide cabinet** | **96%** | [86.5, 98.9] |
| hinge cabinet | 68% | [54.2, 79.2] |
| top burner | 54% | [40.4, 67.0] |
| bottom burner | 8% | [3.2, 18.8] |
| kettle | 0% | [0.0, 7.1] |
| light switch | 0% | [0.0, 7.1] |

`slide cabinet` sits at 96%, and nothing had ever evaluated it. For eleven and
a half hours we watched a three-task average oscillate around 20% and concluded
the policy was marginal, while it was near-solving a fourth task outside the
set.

<div class="finding">
<span class="note-label">Three lessons, in order of cost</span>
<ol>
<li><strong>Compute the variance of your metric before you trust a reading of
it.</strong> One line of arithmetic would have invalidated the overnight
protocol before it ran.</li>
<li><strong>Evaluate on everything you train on.</strong> A subset is a
hypothesis about which tasks matter, and ours was wrong about the best one.</li>
<li><strong>Selecting checkpoints is a different job from monitoring
training.</strong> In-loop evaluation should be a liveness check. Ranking
belongs offline, at a sample size chosen from the effect you want to see.</li>
</ol>
</div>

None of this was expensive. 1,050 rollouts took about 25 minutes on one GPU.
The measurement that would have prevented a wasted night cost less than the
night did.
