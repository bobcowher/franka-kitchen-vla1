---
title: "The action head"
part: "Part II · The pieces"
chapter: 8
weight: 8
standfirst: "Two LayerNorms and a Linear. The LayerNorms are not optional, and finding out why meant throwing away the first diagnostic."
---

Here is the entire trained component of the system.

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

The rest of this chapter is why those three lines are what they are.

## The diagnostic that lied

First question, before anything else: does the image change the hidden state?
If ten different frames produce the same vector, the head has nothing to read
and no amount of training helps.

The obvious check is cosine similarity between hidden states from different
frames. It reads **0.993 to 0.999**.

That looks like total failure. It is wrong.

<div class="trap">
<span class="note-label">Trap · massive activations</span>
SmolLM2's residual stream carries a few dimensions of enormous magnitude. One
of them, dimension 232, has <code>|mean| = 34.0</code>, roughly sixty times the
median dimension. These dominate every dot product, so cosine similarity
between any two hidden states in this model reads ~0.99 whether or not the
input mattered. The instrument was measuring the outliers, not the signal.
</div>

Stop using an angle. Use a decomposition. Write `h = mean + deviation` across
your samples and compare the norms:

| varying | ‖mean‖ | ‖deviation‖ | ratio |
|---|---|---|---|
| ten frames, one instruction | 53.8 | 2.77 | 0.0515 |
| one frame, seven instructions | 53.1 | 6.50 | 0.1225 |

Both signals are real. The instruction moves the hidden state 2.4× harder than
the image does, so language conditioning is alive before a single gradient
step.

## Why that ratio demands a norm

0.0515 has a direct consequence. The vector your head reads is about **95% a
fixed offset** and 5% the part that varies with the input.

A bare `Linear` can represent the answer; the bias absorbs the constant. But
gradient descent on it is miserable, because the gradient along the directions
that carry information is scaled by their tiny magnitude relative to that
constant. The symptom is that overfitting ten samples, which should be
trivial, stalls.

| head | lr | loss after 400 steps |
|---|---|---|
| Linear | 1e-3 | 0.007525 |
| Linear | 1e-2 | 0.000388 |
| LayerNorm → Linear | 1e-3 | 0.010186 |
| **LayerNorm → Linear** | **1e-2** | **0.000000** |
| LayerNorm → 512 → ReLU → 9 | 1e-3 | 0.000022 |

`LayerNorm` centers and rescales each sample, putting the 5% that varies on
equal footing with the 95% that never does.

**Two norms rather than one**, because the two streams have genuinely different
scales. Pooling 64 vectors averages away much of the outlier structure that the
single last token still carries at full magnitude. One shared norm would have
to compromise between them.

## No tanh

The behavior-cloning baseline this project grew out of had a `tanh` on the
output, which is reasonable given an action space of `[−1, 1]`.

Leave it out. The gripper dimensions are exactly ±1 in 100% of demonstration
steps, and `tanh` reaches ±1 only in the limit, so exact zero loss becomes
unreachable and [the gate]({{< relref "chapters/11-the-gate" >}}) loses its only unambiguous
reading. The environment clips to the valid range anyway, so the nonlinearity
buys nothing and costs a diagnostic.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
<code>sum(p.numel() for p in head.parameters())</code> should print 21,129:
two norms at 1,920 parameters each, and a 1,920 × 9 linear with bias.
</div>
