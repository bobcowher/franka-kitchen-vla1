"""The gate for the design: can a Linear on the last prefix token fit at all?

Three questions, cheapest first, because a failure in an earlier one makes the
later ones meaningless:

  1. Does the last token's hidden state change when the image changes? If the
     ten states are all the same vector, the head has nothing to read and no
     amount of training will help. This is the claim the whole architecture
     rests on -- that causal attention has already carried the image forward to
     the final position -- and it was inferred, not measured.
  2. Does it change when the instruction changes? That is language conditioning
     working at all.
  3. Can the head drive MSE on ten real samples to ~zero? Proves shapes line up
     and gradient reaches the weights. Proves nothing about generalization.

Writes no checkpoint.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

import frames
from model import Model
from tasks import TASK_DESCRIPTIONS, task_index

N = 10
STEPS = 400
IMAGE_SIZE = 448

shard = sorted(glob.glob("dataset/microwave/*.npz"))[0]
data = np.load(shard)

# Spread across the episode rather than the first ten steps, which are nearly
# the same picture and would make question 1 look worse than it is.
index = np.linspace(0, len(data["action"]) - 1, N).astype(int)
images = frames.resize(data["camera_scene"][index], IMAGE_SIZE)
actions = torch.tensor(data["action"][index], dtype=torch.float32).cuda()
description = str(data["task_description"])
task = torch.tensor(task_index(description)).repeat(N).cuda()

print(f"{shard}: {description!r}, steps {index.tolist()}")

model = Model(num_actions=actions.shape[1], name="overfit_probe").cuda()


def hidden(frames_u8, task_ids):
    """The vector the head sees, before the head."""
    with torch.no_grad():
        out = model.vlm(input_ids=model.prompt_ids[task_ids],
                        attention_mask=model.prompt_mask[task_ids],
                        pixel_values=model.preprocess(frames_u8))
    return out.last_hidden_state[torch.arange(len(task_ids), device="cuda"),
                                 model.prompt_end[task_ids]].float()


def spread(h, label):
    """How much of the hidden state actually varies across these inputs.

    Not cosine similarity. SmolLM2's residual stream has massive activation
    outliers -- one dim here carries sixty times the median magnitude -- so
    every pair reads ~0.99 whether or not the input mattered. Split
    h = mean + deviation and report the ratio instead; that number is small
    but real, and it is what the head's LayerNorm exists to rescale.
    """
    deviation = (h - h.mean(0)).norm(dim=1).mean()
    print(f"  {label}: ||mean|| {h.mean(0).norm():.1f}  "
          f"||deviation|| {deviation:.2f}  ratio {deviation / h.mean(0).norm():.4f}")


print("\n1. ten different frames, same instruction")
spread(hidden(images, task), "hidden state")

print("\n2. one frame, all seven instructions")
all_tasks = torch.arange(len(TASK_DESCRIPTIONS)).cuda()
one_frame = np.repeat(images[:1], len(TASK_DESCRIPTIONS), axis=0)
spread(hidden(one_frame, all_tasks), "hidden state")

print(f"\n3. overfitting {N} samples for {STEPS} steps")
optimizer = torch.optim.Adam(model.head.parameters(), 1e-2)
for step in range(STEPS):
    loss = F.mse_loss(model(images, task), actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == STEPS - 1:
        print(f"  step {step:>4}  loss {loss.item():.6f}")

print(f"\n  target variance (loss if it predicted the mean): "
      f"{actions.var(0, unbiased=False).mean().item():.6f}")
