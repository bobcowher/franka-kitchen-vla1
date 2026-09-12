---
title: "Reading the prefix"
part: "Part II · The model"
chapter: 5
weight: 5
standfirst: "Which hidden states the head looks at is an architecture decision, not a detail. Causal attention forces the answer."
---

The VLM hands back one 960-dimensional vector per input position — 79 to 84 of
them. The head needs a fixed-size input. Which positions do you read?

The three obvious candidates:

1. **The last token.** It has attended to everything before it.
2. **Mean-pool the image tokens.** They hold the visual content.
3. **Both.**

## The argument from causal attention

Look at the token order. The chat template puts the image *before* the text:

<div class="tokens">
  <span class="tok-img">64 image tokens</span>
  <span class="tok-txt">instruction</span>
  <span class="tok-read">last</span>
</div>
<p class="caption">Attention is causal — each position sees only what precedes it.</p>

This has a consequence that settles the design:

<div class="finding">
<span class="note-label">Consequence</span>
<p><strong>The image tokens cannot see the instruction.</strong> They come
first, and attention runs left to right. Pooling them gives you a
representation of the scene that is <em>identical for all seven tasks</em> —
task-blind by construction.</p>
<p><strong>The last token can see everything</strong>, but it is one vector
that has to summarize 64 image positions through a bottleneck it was
pretrained to use for predicting the next word, not for describing geometry.</p>
</div>

So the two candidates are not competing views of the same information. One is
language-aware and visually compressed; the other is visually detailed and
language-blind. Reading both is the only option that has all the information
in it.

## What the measurement says

Reasoning is nice. The frozen VLM makes this cheap to actually test — encode
a sample once, then fit head variants on the cached vectors in seconds. The
harness is [Chapter 9](../09-the-fast-loop/).

Held-out weighted MSE, held out by shard, five seeds each. Predicting the
dataset mean scores 0.2132.

| head | params | mean | best | worst |
|---|---|---|---|---|
| last token only | 10,569 | 0.1090 | 0.1025 | 0.1148 |
| pooled image only | 10,569 | 0.0972 | 0.0930 | 0.1030 |
| **fused (shipped)** | **21,129** | **0.0905** | **0.0885** | **0.0934** |
| fused → 512 → 9 | 992,009 | 0.1519 | 0.1057 | 0.2133 |

Three things fall out of that table.

**The architectural argument was right.** Fusing beats either stream alone,
and it also has the tightest spread across seeds — it is the most *stable*
choice, not just the best average.

**Pooled beats last, alone.** Mildly surprising given that pooled is
task-blind. It says the visual detail is worth more than the language
conditioning on this particular dataset, which is a comment on the dataset
(seven tasks, heavily visually distinguishable) more than on the method.

**Capacity is not the bottleneck.** A head with 47× more parameters is
*worse*, and wildly unstable — its worst seed is exactly the
predict-the-mean baseline, meaning it sometimes learns nothing at all. The
frozen prefix is the ceiling here, and no amount of head makes it higher.
That single row is why the next lever in this project is unfreezing rather
than a bigger head.

## The implementation

```python
h = self.vlm(input_ids=self.prompt_ids[task],
             attention_mask=self.prompt_mask[task],
             pixel_values=self.preprocess(frames)).last_hidden_state.float()

last = h[torch.arange(len(task)), self.prompt_end[task]]
mask = self.image_positions[task]
pooled = (h * mask).sum(1) / mask.sum(1)

return self.head(last, pooled)
```

The masked mean is written out rather than using a helper because the mask
has to broadcast over the feature dimension and the denominator has to be the
token count, not the element count. Both are easy to get subtly wrong and
neither would raise.
