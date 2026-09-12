---
title: "The action head"
part: "Part II · The model"
chapter: 6
weight: 6
standfirst: "Two LayerNorms and a Linear. The LayerNorms are not optional, and finding out why required throwing away the first diagnostic."
---

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

That is the whole thing. The interesting part is why the norms are there, and
that story starts with a diagnostic that lied.

## The diagnostic that lied

First question before anything else: **does the image actually change the
hidden state?** If ten different frames produce the same vector, the head has
nothing to read and no amount of training will help.

The obvious check is cosine similarity between hidden states across different
frames. It read **0.993 to 0.999**.

That looks like total failure — the image not getting through at all. It sent
the build down a dead end for a while. It is also wrong.

<div class="trap">
<span class="note-label">Trap · massive activations</span>
SmolLM2's residual stream carries a handful of dimensions with enormous
magnitude — one of them, dimension 232, has <code>|mean| = 34.0</code>, about
sixty times the median dimension. These dominate every dot product. Cosine
similarity between any two hidden states in this model reads ~0.99 whether or
not the input mattered. The instrument was measuring the outliers, not the
signal.
</div>

The fix is to stop using an angle and start using a decomposition. Write
`h = mean + deviation` across the samples and compare norms:

| varying | ‖mean‖ | ‖deviation‖ | ratio |
|---|---|---|---|
| ten frames, one instruction | 53.8 | 2.77 | 0.0515 |
| one frame, seven instructions | 53.1 | 6.50 | 0.1225 |

Both signals are real. And the instruction moves the hidden state **2.4×
harder than the image does** — language conditioning is alive before a single
gradient step, which is worth knowing before you spend an afternoon wondering
whether it works.

## Why that means the head needs a norm

That `0.0515` has a direct consequence. The vector the head reads is roughly
**95% a fixed offset** and 5% the part that varies with the input.

A bare `Linear` can represent the answer — the bias absorbs the constant. But
gradient descent on it is miserable, because the gradient in the directions
that carry information is scaled by their tiny magnitude relative to the
constant. The symptom is that overfitting ten samples, which should be
trivial, stalls.

| head | lr | loss after 400 steps |
|---|---|---|
| Linear | 1e-3 | 0.007525 |
| Linear | 1e-2 | 0.000388 |
| LayerNorm → Linear | 1e-3 | 0.010186 |
| **LayerNorm → Linear** | **1e-2** | **0.000000** |
| LayerNorm → 512 → ReLU → 9 | 1e-3 | 0.000022 |

Centering and rescaling per sample puts the 5% that varies on equal footing
with the 95% that never does.

**Two norms, not one**, because the last-token stream and the pooled-image
stream have genuinely different scales — pooling 64 vectors averages away a
lot of the outlier structure that the single last token still carries at full
magnitude. One shared norm would have to compromise between them.

## No tanh

The behavior-cloning baseline this project grew out of had a `tanh` on the
output, which is reasonable: the action space is `[−1, 1]`.

It is omitted here deliberately. The gripper dimensions are exactly ±1 in
100% of demonstration steps, and `tanh` reaches ±1 only in the limit — so
exact zero loss becomes unreachable, and [the gate](../08-the-gate/) loses
the only unambiguous reading it has. The environment clips to the valid range
anyway, so the nonlinearity buys nothing and costs a diagnostic.
