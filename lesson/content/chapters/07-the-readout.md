---
title: "The readout"
part: "Part II · The pieces"
chapter: 7
weight: 7
standfirst: "Which hidden states the head reads is an architecture decision, and causal attention forces the answer."
---

The backbone hands us one 960-number vector per input position, so between 79
and 84 of them. We'll call that whole collection the *prefix*: the model's
representation of everything we gave it, sitting where generation would begin if
we were asking for text.

Our head needs a fixed-size input, so we have to choose which positions to read.
There are three obvious candidates:

1. The last token, which has attended to everything before it.
2. The mean of the 64 image tokens, which hold the visual content.
3. Both.

## What the token order forces

Let's look again at how the chat template arranged things. The image comes
first, then the instruction:

<div class="tokens">
  <span class="tok-img">64 image tokens</span>
  <span class="tok-txt">instruction</span>
  <span class="tok-read">last</span>
</div>
<p class="caption">Attention runs left to right, so each position sees only what precedes it.</p>

If attention is causal and the image comes before the text, then the image
positions cannot possibly have seen the instruction. Pooling them should give us
a representation of the scene that is *identical* regardless of which task we
asked for.

That's a strong claim, and it's cheap to check. Let's run the same frame through
with two different task ids and compare both readouts:

```python
frame = np.random.randint(0, 255, (1, 448, 448, 3), dtype=np.uint8)
for t in (3, 6):                      # microwave, hinge cabinet
    task = torch.tensor([t]).cuda()
    out = model.vlm(input_ids=model.prompt_ids[task],
                    attention_mask=model.prompt_mask[task],
                    pixel_values=model.preprocess(frame)).last_hidden_state.float()
    last = out[torch.arange(1).cuda(), model.prompt_end[task]]
    mask = model.image_positions[task]
    pooled = (out * mask).sum(1) / mask.sum(1)
```

<div class="output"><p class="output-label">Comparing the two runs prints</p>

```text
pooled max abs difference  0.000000
last   max abs difference  5.3125
```
</div>

Exactly zero. Not approximately zero, not small: the pooled image representation
is bit-identical for two different instructions, because those 64 positions were
computed before the model had read a single word of the task. The last token,
meanwhile, differs by 5.31.

So our two candidates are not two views of the same information. One is
visually detailed and completely task-blind. The other is task-aware but has
squeezed 64 image positions through a single vector that the model was
pretrained to use for guessing the next word, not for describing geometry.
Reading both is the only option that has all of it.

## Checking the argument against a measurement

Reasoning is cheap and can be wrong, so let's put numbers on it. Because the
backbone is frozen we can encode a sample once and then fit head variants on the
cached vectors in seconds; Chapter 12 builds that harness.

The numbers below are mean squared error on *held-out* data, meaning episodes
the head never trained on. Predicting the dataset mean scores 0.2132, so that's
the number to beat, and each variant is fitted from five different random seeds.

<div class="output"><p class="output-label"><code>python scripts/probe.py</code> prints</p>

```text
head                              params     mean     best    worst
last only                         10,569   0.1090   0.1025   0.1148
pooled only                       10,569   0.0972   0.0930   0.1030
fused                             21,129   0.0905   0.0885   0.0934
fused -> 512 -> 9                992,009   0.1519   0.1057   0.2133
```
</div>

Three things come out of that table.

**The architectural argument holds.** Fusing beats either stream on its own, and
it also has the tightest spread across seeds, so it's the most stable choice as
well as the best one.

**Pooled beats last on its own**, which is mildly surprising given that pooled is
task-blind. It says the visual detail is worth more than the language
conditioning on this particular dataset, which is a comment on our seven
visually distinct tasks rather than on the method.

**Capacity is not what's holding us back.** The head with 47 times more
parameters is *worse*, and unstable enough that its worst seed lands exactly on
the predict-the-mean baseline, meaning that on that seed it learned nothing at
all. The frozen prefix is the ceiling here, and no amount of head raises it.
That single row is the reason Chapter 15 reaches for unfreezing rather than for
a bigger head.

## Writing it

<p class="listing">Listing 7.1 <em>Reading both streams out of the prefix</em></p>
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

`.float()` brings us out of bfloat16 so the head trains in full precision. The
gather on `last` uses `torch.arange` for the batch index and `prompt_end` for the
position, picking one token per row at a different offset in each.

The masked mean is written out longhand rather than pulled from a helper because
two things are easy to get subtly wrong and neither of them raises: the mask has
to broadcast across the feature dimension, and the denominator has to be the
number of image *tokens*, not the number of elements.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
Run the comparison from earlier in this chapter on your own model. You want
<code>pooled</code> identical across two task ids and <code>last</code> clearly
different. If <code>pooled</code> differs, your image mask is picking up text
positions. If <code>last</code> doesn't, the instruction isn't reaching the
model at all, and everything after this point will be a seven-way guess.
</div>

Next: the action head.
