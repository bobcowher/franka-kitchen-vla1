---
title: "The readout"
part: "Part II · The pieces"
chapter: 7
weight: 7
standfirst: "Which hidden states the head looks at is an architecture decision. Causal attention forces the answer."
---

The VLM returns one 960-dimensional vector per input position, 79 to 84 of
them. This is the <em class="term">prefix</em>: the model's representation of
everything you handed it, before any generation would begin.

Your head needs a fixed-size input. Which positions do you read?

Three candidates:

1. The last token, which has attended to everything before it.
2. The mean of the image tokens, which hold the visual content.
3. Both.

## The argument from causal attention

Look at the order the chat template produces. Image first, then text:

<div class="tokens">
  <span class="tok-img">64 image tokens</span>
  <span class="tok-txt">instruction</span>
  <span class="tok-read">last</span>
</div>
<p class="caption">Attention runs left to right. Each position sees only what precedes it.</p>

<div class="finding">
<span class="note-label">Consequence</span>
<p><strong>The image tokens cannot see the instruction.</strong> They come
first, and attention is causal. Pooling them gives a representation of the
scene that is identical for all seven tasks — task-blind by construction.</p>
<p><strong>The last token can see everything</strong>, but it is one vector
summarizing 64 image positions through a bottleneck the model was pretrained to
use for predicting the next word, not for describing geometry.</p>
</div>

So these are not two views of the same information. One is language-aware and
visually compressed. The other is visually detailed and language-blind. Reading
both is the only option that has all of it.

## What the measurement says

Reasoning is cheap. Measure it. Because the VLM is frozen you can encode a
sample once and fit head variants on the cached vectors in seconds; that
harness is [Chapter 12]({{< relref "chapters/12-the-fast-loop" >}}).

Numbers below are mean squared error on <em class="term">held-out</em> data —
episodes the head never trained on. Predicting the dataset mean scores 0.2132,
so that is the number to beat. Five random seeds per variant.

| head | params | mean | best | worst |
|---|---|---|---|---|
| last token only | 10,569 | 0.1090 | 0.1025 | 0.1148 |
| pooled image only | 10,569 | 0.0972 | 0.0930 | 0.1030 |
| **fused (shipped)** | **21,129** | **0.0905** | **0.0885** | **0.0934** |
| fused → 512 → 9 | 992,009 | 0.1519 | 0.1057 | 0.2133 |

Three things fall out.

**The architectural argument holds.** Fusing beats either stream alone, and has
the tightest spread across seeds. It is the most stable choice as well as the
best.

**Pooled beats last, on its own.** Mildly surprising, since pooled is
task-blind. It says visual detail is worth more than language conditioning on
this dataset, which is a comment on the dataset — seven visually distinct
tasks — more than on the method.

**Capacity is not the bottleneck.** A head with 47× more parameters is worse,
and unstable enough that its worst seed lands exactly on the
predict-the-mean baseline, meaning it sometimes learns nothing. The frozen
prefix is the ceiling, and no head makes it higher. That single row is why the
next lever in this project is unfreezing rather than a bigger head.

## The code

<p class="filename">Filename: <strong>model.py</strong></p>

```python
def forward(self, frames, task):
    task = task.reshape(-1).to(self.device)

    h = self.vlm(input_ids=self.prompt_ids[task],
                 attention_mask=self.prompt_mask[task],
                 pixel_values=self.preprocess(frames)).last_hidden_state.float()

    # The image precedes the text and attention is causal, so image
    # positions cannot see the instruction and the last token can. Need both.
    last = h[torch.arange(len(task), device=self.device), self.prompt_end[task]]
    mask = self.image_positions[task]
    pooled = (h * mask).sum(1) / mask.sum(1)

    return self.head(last, pooled)
```

`.float()` converts out of bfloat16, so the head trains in full precision.

The gather on `last` uses `torch.arange` for the batch index and `prompt_end`
for the position index, picking one token per row at a different offset in each.

The masked mean is written out rather than pulled from a helper because two
things are easy to get subtly wrong and neither raises: the mask has to
broadcast over the feature dimension, and the denominator has to be the token
count, not the element count.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
Feed the same frame with two different task ids. <code>pooled</code> should be
identical for both, and <code>last</code> should differ. If <code>pooled</code>
differs you have the token order wrong; if <code>last</code> does not, the
instruction is not reaching the model.
</div>
