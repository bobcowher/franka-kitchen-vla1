---
title: "Rollouts, and how the eval lied"
part: "Part IV · Knowing whether it works"
chapter: 10
weight: 10
standfirst: "The architecture worked long before we could tell. Three separate failures, all of them in the measurement."
---

Training loss is not the objective. The objective is whether the arm opens the
cabinet. So the training loop stops every 2,500 epochs and runs rollouts:
three tasks, three attempts each, and logs the mean success rate.

Three rollouts per task was a deliberate, correct decision at the start of the
project — rollouts are slow, early iteration matters more than precision, and
any success above zero was the win condition. It was right on day one.

It quietly stopped being right, and nothing announced the transition.

## The arithmetic nobody ran

`eval/mean` is the average of three independent `Binomial(3, p)/3` estimates.
Its standard deviation is `√(p(1−p)/9)`.

At a true success rate of `p = 0.35`, that is **0.159** — a 95% interval of
roughly **±31 percentage points**.

<div class="finding">
<span class="note-label">The consequence</span>
An overnight run produced ~40 evaluation points that swung between 0% and 56%,
and we read that curve for hours looking for trends — a peak at 40K, a
collapse after it, a recovery. <strong>The entire pattern is what a single
unchanging policy looks like when you sample it forty times at that
resolution.</strong> There was never a trend to find.
</div>

## Measuring it properly

The fix needs no new ideas, just more samples. `scripts/evaluate.py` loads any
checkpoint and runs it through the *same* `agent.test()` code path at
arbitrary `n`, with a Wilson interval on the result. Rollouts parallelize
almost perfectly — each one spends most of its wall clock waiting on the
physics simulator, not the GPU — so seven at a time on one card is nearly free.

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

`e100000` is the final checkpoint — the thing you get if you train for eleven
hours and keep the weights at the end.

It scored **0 successes in 100 rollouts.** The interval tops out at 3.7%, so
this is not an unlucky sample; the policy genuinely collapsed somewhere in the
last 2,500 epochs.

Meanwhile `train/loss` spent that entire window in its best-ever region,
0.04–0.09, declining smoothly. **The loss curve reported nothing.**

<div class="trap">
<span class="note-label">Trap</span>
In behavior cloning, training loss and task success are only loosely coupled,
and they can decouple completely. Any pipeline that keeps final weights is one
bad window away from shipping a policy that does nothing — with a training
curve that looks perfect. Keep the best evaluated checkpoint, and evaluate
often enough that "best" means something.
</div>

## Failure two: the ranking was wrong

`e45000` and `e70000` both read 44% on their in-run evaluation. They are 60%
and 35% — non-overlapping intervals, a real 25-point gap that nine rollouts
could not see.

Every decision made against that ranking overnight was made against noise.

There is a subtler finding in the per-task columns. The checkpoints are not
getting uniformly better or worse — they are **trading tasks off against each
other**:

| checkpoint | hinge cabinet | top burner |
|---|---|---|
| e45000 | 86% | 34% |
| e40000 | 68% | 54% |
| e97500 | 26% | 68% |

The mean stays roughly flat while the policy drifts sideways, swapping which
task it is good at. That is what interference looks like when 21,129
parameters are asked to hold several behaviors over a frozen representation —
and it is a far better argument for unfreezing than any trend in `eval/mean`
ever was.

## Failure three: we never looked at the best task

The evaluation set was three tasks. The model is trained on **seven**. Four
had never been rolled out, not once, in any run.

The first question asked about the eval set — *why are we testing microwave
when we know it never succeeds?* — was the thread that unravelled this. It
was a question about wasted compute. The answer turned out to be much worse
than waste.

| task | rate | 95% CI |
|---|---|---|
| **slide cabinet** | **96%** | [86.5, 98.9] |
| hinge cabinet | 68% | [54.2, 79.2] |
| top burner | 54% | [40.4, 67.0] |
| bottom burner | 8% | [3.2, 18.8] |
| kettle | 0% | [0.0, 7.1] |
| light switch | 0% | [0.0, 7.1] |

`slide cabinet` is at **96%**, and nothing had ever evaluated it. For eleven
and a half hours we watched a three-task average oscillate around 20% and
concluded the policy was marginal, while it was quietly near-solving a fourth
task that wasn't in the set.

<div class="finding">
<span class="note-label">Three lessons, in order of cost</span>
<ol>
<li><strong>Compute the variance of your metric before you trust a reading of
it.</strong> One line of arithmetic would have invalidated the entire
overnight protocol before it ran.</li>
<li><strong>Evaluate on everything you train on.</strong> A subset is a
hypothesis about which tasks matter, and ours was wrong about the best
one.</li>
<li><strong>Selecting checkpoints is a different job from monitoring
training.</strong> In-run eval should be a liveness check. Ranking belongs
offline, at a sample size chosen from the effect you are trying to see.</li>
</ol>
</div>

## Rollouts are cheap, and that is the punchline

The most uncomfortable part of this chapter is that none of it was expensive.
1,050 rollouts took about 25 minutes on one GPU, because they parallelize and
because a failed episode is bounded at 400 steps.

The measurement that would have prevented a wasted night cost less than the
night did.
