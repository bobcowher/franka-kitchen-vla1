---
title: "The model, assembled"
part: "Part II · The pieces"
chapter: 9
weight: 9
standfirst: "Every piece from Chapters 5 to 8 in one file, plus the preprocessing step that made training seven times faster."
---

You now have the parts. This chapter puts them in one file and adds the last
missing piece.

## Preprocessing, on the GPU

The obvious worry about running a 500M model in the training loop is that the
forward pass is too slow. It isn't. Profiling found that **85% of step time was
CPU-side image preprocessing** inside the HuggingFace processor.

| path, batch 64 | processor | gpu | total | rate |
|---|---|---|---|---|
| `processor(...)` | 827 ms | 140 ms | 968 ms | 1.0 it/s |
| **gpu preprocessing** | — | 144 ms | 144 ms | **7.0 it/s** |

Read the image processor's config and its entire image branch turns out to be
`resize → ×1/255 → (x − 0.5)/0.5`. Nothing else. Reproduce it in three lines of
torch:

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

The `permute` is where HWC becomes the NCHW that `interpolate` expects, which
is why [Chapter 3]({{< relref "chapters/03-the-environment" >}}) could leave frames in HWC. The
`frames.dim() == 3` branch lets a single rollout observation through without a
batch dimension. The final `unsqueeze(1)` adds the sub-image axis SmolVLM2
expects, which is 1 now that splitting is off.

<div class="trap">
<span class="note-label">Trap · one preprocessing path, not two</span>
This function serves both training and rollout. If training normalizes on the
GPU and rollout calls the processor, any drift between them produces a policy
that trains fine and fails in the environment, with nothing to point at. One
function, both callers.
</div>

A 7× speedup from deleting a library call. The instinct was to shrink the VLM;
the fix never touched it. **Profile the step before optimizing the model.**

## The whole file

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

`device` reads off the head rather than the VLM because the head is the part
you own, and it is always on whatever device you moved the model to.

Checkpoints save **only the head**. The VLM is frozen and reloadable from the
hub, so storing it would write 1 GB per checkpoint to record 21,129 numbers
that changed. A head checkpoint is 85 KB, which is what makes it practical to
keep a snapshot at every evaluation — and [Chapter 13]({{< relref "chapters/13-rollouts" >}}) shows
why that matters more than it sounds.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
Construct the model and run one batch of random uint8 frames through it. You
should get back <code>(batch, 9)</code> float32, and it should not raise. If
you get a channel-count error, your frames are in the wrong axis order.
</div>
