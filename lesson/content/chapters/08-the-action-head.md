---
title: "The action head"
part: "Part II · The pieces"
chapter: 8
weight: 8
standfirst: "Two LayerNorms and a Linear. The norms are not optional, and working out why meant throwing away our first diagnostic."
---

Here is the entire trained component of the system:

<p class="listing">Listing 8.1 <em>The action head</em></p>
<p class="filename">Filename: <strong>model.py</strong></p>

```python
class ActionHead(nn.Module):
    """Separate norms because the two streams have very different scales."""

    def __init__(self, hidden, num_actions):
        super().__init__()
        self.norm_last = nn.LayerNorm(hidden)
        self.norm_pooled = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden * 2, num_actions)

    def forward(self, last, pooled):
        return self.out(torch.cat([self.norm_last(last),
                                   self.norm_pooled(pooled)], dim=1))
```

<div class="output"><p class="output-label">Printing its parameters gives</p>

```text
head parameters  21,129
  norm_last.weight       (960,)
  norm_last.bias         (960,)
  norm_pooled.weight     (960,)
  norm_pooled.bias       (960,)
  out.weight             (9, 1920)
  out.bias               (9,)
```
</div>

The `LayerNorm`s are not optional. Here's why.

## Measuring what the head actually sees

The first question to answer: does the hidden state actually change when the
image changes? If ten different frames produce the same vector, the head has
nothing to read.

Cosine similarity is the obvious check. On ten frames from the same episode:

<div class="output"><p class="output-label">This prints</p>

```text
cosine similarity    min 0.9978  max 0.9996
```
</div>

That looks like total failure. Ten genuinely different pictures, and the model's
output barely moves. The instrument is broken, not the model.

Here's what's going on. Let's look at the mean hidden state across those ten
frames, dimension by dimension:

<div class="output"><p class="output-label">This prints</p>

```text
largest |mean| dim   232  value 26.5
median |mean|        0.475
ratio largest/median 56x
```
</div>

<div class="trap">
<span class="note-label">Trap · massive activations</span>
SmolLM2's residual stream carries a handful of dimensions with enormous
magnitude. Dimension 232 here is <strong>fifty-six times</strong> the median
dimension, and it is the same 26.5 whatever image you show the model. Those few
dimensions dominate every dot product, so cosine similarity between any two
hidden states in this model reads about 0.99 whether or not the input mattered.
We were measuring the outliers, not the signal.
</div>

The fix is to stop measuring an angle and start measuring a magnitude. Split the
hidden state into the part that's constant across samples and the part that
varies, then compare their sizes:

```python
deviation = (h - h.mean(0)).norm(dim=1).mean()
ratio = deviation / h.mean(0).norm()
```

<div class="output"><p class="output-label">On real demonstration frames this gives</p>

```text
1. ten different frames, same instruction
  hidden state: ||mean|| 53.8  ||deviation|| 2.77  ratio 0.0515

2. one frame, all seven instructions
  hidden state: ||mean|| 53.1  ||deviation|| 6.50  ratio 0.1225
```
</div>

Both signals are real after all. And notice the second line: the instruction
moves the hidden state 2.4 times harder than the image does, so language
conditioning is working before we've taken a single gradient step.

## Why that ratio demands a norm

That 0.0515 has a consequence we have to design around. The vector our head
reads is roughly **95% a fixed offset** and only 5% the part that varies with
what the model is looking at.

A bare `Linear` can represent the right answer, since its bias absorbs the
constant. But gradient descent on it is miserable, because the gradient along the
directions that carry information is scaled by their tiny magnitude relative to
that enormous constant. The symptom is that overfitting ten samples, which ought
to be trivial, stalls instead.

Here is what a bare `Linear` head scores against the normed version:

| head | lr | loss after 400 steps |
|---|---|---|
| Linear | 1e-3 | 0.007525 |
| Linear | 1e-2 | 0.000388 |
| LayerNorm → Linear | 1e-3 | 0.010186 |
| **LayerNorm → Linear** | **1e-2** | **0.000000** |
| LayerNorm → 512 → ReLU → 9 | 1e-3 | 0.000022 |

`LayerNorm` centres and rescales each sample, which puts the 5% that varies on
equal footing with the 95% that never does.

**Two norms rather than one** because the two streams genuinely differ in scale.
Averaging 64 image vectors washes out much of the outlier structure that the
single last token still carries at full strength, so one shared norm would have
to compromise between them.

## Why there's no tanh

The behavior-cloning policy this project grew out of had a `tanh` on its output,
which is a reasonable thing to do when the action space is `[−1, 1]`.

We're leaving it out on purpose. The gripper dimensions are exactly ±1 in every
demonstration step, and `tanh` only reaches ±1 in the limit, so exact zero loss
would become unreachable. Chapter 11 is built entirely around a test that asks
whether the loss can hit exactly zero, and we'd rather not blunt our only
unambiguous diagnostic to gain a bound the environment already enforces by
clipping.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
<code>sum(p.numel() for p in head.parameters())</code> should print
<strong>21,129</strong>: two norms at 1,920 parameters each, plus a 1,920 × 9
linear with its bias.
</div>

<div class="exercise">
<h4>Exercise 8.1 &nbsp;Find the outlier dimensions yourself</h4>
<p>Encode twenty frames, take the mean hidden state, and sort the dimensions by
absolute magnitude. How many account for half the total? Then zero out the top
five and recompute cosine similarity between frames.</p>
<p>If cosine suddenly becomes informative, you've demonstrated the failure
directly rather than taking our word for it, and you'll recognize it instantly
the next time a similarity metric reads 0.99 on a model that's working fine.</p>
</div>

Next: assembling the model.
