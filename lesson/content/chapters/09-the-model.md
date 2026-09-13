---
title: "The model, assembled"
part: "Part II · The pieces"
chapter: 9
weight: 9
standfirst: "Every piece from Chapters 5 to 8 in one file, plus the discovery that our 460M-parameter model was never the slow part."
---

We have all the pieces now, so let's put them in one file. There's one thing
still missing, and finding it is the most useful hour in this chapter.

## The bottleneck is not where you expect

The obvious worry about running a 460-million-parameter model inside the
training loop is that the forward pass will be too slow to iterate on. So before
committing to the design, we profiled a single training step at batch 64.

| path, batch 64 | processor | gpu | total | rate |
|---|---|---|---|---|
| `processor(...)` | 827 ms | 140 ms | 968 ms | 1.0 it/s |
| **gpu preprocessing** | — | 144 ms | 144 ms | **7.0 it/s** |

The model was never the problem. **Eighty-five percent of every step** was
CPU-side image preprocessing inside the HuggingFace processor, resizing and
normalizing 64 frames one at a time while a perfectly good GPU sat waiting.

So we read the image processor's configuration to see what it actually does to
an image, and the entire branch turns out to be three operations: resize to 512,
divide by 255, then subtract 0.5 and divide by 0.5. Nothing else. All three are
trivial on a GPU.

<p class="listing">Listing 9.1 <em>The processor's image branch, reimplemented on the GPU</em></p>
<p class="filename">Filename: <strong>model.py</strong></p>

```python
def preprocess(self, frames):
    """uint8 (B,H,W,3) -> pixel_values. Same path for training and rollout.

    This is the processor's image branch, on the GPU. Calling the processor
    instead costs 85% of the step.
    """
    if not torch.is_tensor(frames):
        frames = torch.from_numpy(np.ascontiguousarray(frames))
    if frames.dim() == 3:
        frames = frames[None]

    x = frames.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255
    x = nn.functional.interpolate(x, size=VLM_IMAGE_SIZE, mode="bilinear",
                                  align_corners=False)
    return ((x - 0.5) / 0.5).to(torch.bfloat16).unsqueeze(1)
```

<div class="output"><p class="output-label">Feeding it a batch of 64 frames gives</p>

```text
preprocess -> (64, 1, 3, 512, 512) torch.bfloat16
```
</div>

Four things are happening in those three lines. The `permute` is where HWC
becomes the channels-first layout the model wants, which is exactly why
Chapter 3 could leave frames in HWC. The `frames.dim() == 3` branch lets a single
rollout observation through without a batch dimension. The cast to bfloat16
matches the backbone. And the final `unsqueeze(1)` adds the sub-image axis
SmolVLM2 expects, which is 1 now that we turned splitting off.

A sevenfold speedup from deleting a library call. The instinct had been to
reach for a smaller model, and the fix never touched the model at all.

<div class="trap">
<span class="note-label">Trap · one preprocessing path, not two</span>
This function serves both training and rollout. If training normalizes on the
GPU while rollout calls the processor, any drift between the two produces a
policy that trains beautifully and fails in the environment, with nothing to
point at. One function, both callers.
</div>

## The whole file

<p class="listing">Listing 9.2 <em>model.py, assembled</em></p>
<p class="filename">Filename: <strong>model.py</strong></p>

```python
"""Frozen SmolVLM2 with an action head that emits joint velocity."""
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText

from tasks import TASK_DESCRIPTIONS

VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_IMAGE_SIZE = 512   # vision_config.image_size
HIDDEN = 960           # text_config.hidden_size


class Model(nn.Module):

    def __init__(self, num_actions, checkpoint_dir='checkpoints',
                 name='vla_network'):
        super().__init__()

        processor = AutoProcessor.from_pretrained(VLM)
        # On, a frame is 17 sub-images and 1139 tokens. Off, 79-84.
        processor.image_processor.do_image_splitting = False
        self._build_prompts(processor)

        # .model drops the unused 49280 x 960 vocabulary projection.
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            VLM, dtype=torch.bfloat16, attn_implementation="sdpa").model
        self.vlm.requires_grad_(False)

        self.head = ActionHead(HIDDEN, num_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.train()

    @property
    def device(self):
        return next(self.head.parameters()).device

    def save_checkpoint(self):
        torch.save(self.head.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, path=None):
        self.head.load_state_dict(
            torch.load(path or self.checkpoint_file, map_location=self.device))
```

`device` reads off the head rather than the backbone because the head is the
part we own, and it always sits on whatever device we moved the model to.

Checkpoints save **only the head**. The backbone is frozen and can be pulled
from the hub again at any time, so storing it would mean writing a gigabyte per
checkpoint to record 21,129 numbers that changed. A head checkpoint is 85 KB,
which is what makes it practical to keep a snapshot at every single evaluation,
and Chapter 13 is about to show you why that matters far more than it sounds.

## Checking it runs

<div class="output"><p class="output-label">Feeding a batch of 64 random frames through the finished model</p>

```text
model(frames, tasks) -> (64, 9) torch.float32
forward, batch 64: 144 ms
```
</div>

Nine numbers per frame, in float32, at 144 milliseconds a batch. That matches the
GPU-only row in the table at the top of this chapter, which is a decent sign
that nothing has crept back onto the CPU.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
Feed a batch of random <code>uint8</code> frames through and confirm you get
<code>(batch, 9)</code> float32 back without an exception. If you see a
channel-count error, your frames are in the wrong axis order and Chapter 3 is
where to look.
</div>

<div class="exercise">
<h4>Exercise 9.1 &nbsp;Profile it yourself</h4>
<p>Time one step with your own preprocessing and again with
<code>processor(...)</code>, remembering <code>torch.cuda.synchronize()</code>
either side of the call so you're timing the GPU rather than the queue.</p>
<p>Then try batch 8 and batch 128. The ratio between the two paths is not
constant, and working out why tells you something about which of the two is
bound by what.</p>
</div>

Next, we'll write the training loop, which is ordinary behavior cloning with one
weighted term that comes straight out of the statistics in Chapter 4.
