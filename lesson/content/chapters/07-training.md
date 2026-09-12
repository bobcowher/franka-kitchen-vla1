---
title: "The training loop"
part: "Part III · Training"
chapter: 7
weight: 7
standfirst: "Behavior cloning, 56,005 demonstration steps, and the discovery that the GPU was never the bottleneck."
---

Ordinary behavior cloning. Sample a batch of (frame, task, action) triples,
forward, MSE against the demonstrated action, step Adam on the head only.

```python
pred_actions = self.model(states['camera_scene'], tasks)

arm_loss     = F.mse_loss(actions[:, :7], pred_actions[:, :7])
gripper_loss = F.mse_loss(actions[:, 7:], pred_actions[:, 7:])

loss = arm_loss + GRIPPER_WEIGHT * gripper_loss
```

## Why the gripper is weighted down

`GRIPPER_WEIGHT = 0.125`, and it comes from measuring the data rather than
from tuning.

Dimensions 7 and 8 are the gripper. They are ±1 in **100%** of demonstration
steps — never anything in between — with a standard deviation of 0.902
against roughly 0.29 for a typical arm joint. Left unweighted they carry about
8× an arm joint's variance and dominate the gradient, so the model optimizes
the easy binary decision at the expense of the seven continuous ones that
actually position the arm.

<div class="note">
<span class="note-label">Also found in the data</span>
<strong>Dimension 4 is dead.</strong> Standard deviation 0.017, and the 1st and
99th percentiles are both exactly 0.000 — it is zero in over 98% of steps.
Nothing on the gamepad used to collect demonstrations maps to that joint. The
demonstrations are effectively 6 arm degrees of freedom plus a gripper, not 7,
which is worth knowing before drawing conclusions about what poses are
reachable.
</div>

## The bottleneck was the CPU

The obvious worry about running a 500M model in the training inner loop is
that the forward pass is too slow. It wasn't. Profiling the step found that
**85% of the time was CPU-side image preprocessing** in the HuggingFace
processor.

| path, batch 64 | processor | gpu | total | rate |
|---|---|---|---|---|
| `processor(...)` | 827 ms | 140 ms | 968 ms | 1.0 it/s |
| **gpu preprocessing** | — | 144 ms | 144 ms | **7.0 it/s** |

Reading the image processor's config showed that its entire image branch is
`resize → ×1/255 → (x − 0.5)/0.5`. Nothing else. Three lines of torch
reproduce it on the GPU:

```python
x = frames.to(self.device).permute(0, 3, 1, 2).float() / 255
x = F.interpolate(x, size=VLM_IMAGE_SIZE, mode="bilinear", align_corners=False)
return ((x - 0.5) / 0.5).to(torch.bfloat16).unsqueeze(1)
```

A 7× speedup from deleting a library call. The lesson generalizes further than
this project: **profile the step before optimizing the model.** The instinct
was to shrink the VLM; the actual fix never touched it.

<div class="trap">
<span class="note-label">Trap · one preprocessing path, not two</span>
This function is used by both training and rollout. If the training path
normalizes on the GPU and the rollout path calls the processor, any drift
between them produces a policy that trains fine and fails in the environment,
with nothing to point at. One function, both callers.
</div>

## Numbers for budgeting

<dl class="stats">
  <div><dt>Batch</dt><dd>64</dd></div>
  <div><dt>Head LR</dt><dd>1e-3</dd></div>
  <div><dt>Throughput</dt><dd>2.55/s</dd></div>
  <div><dt>VRAM</dt><dd>2.9 GB</dd></div>
  <div><dt>Steps</dt><dd>56,005</dd></div>
</dl>

2.55 epochs/sec on an RTX 3090, so a 100,000-epoch run is about 11 hours of
training plus roughly 100 minutes of in-run evaluation. That evaluation
overhead is going to become the subject of
[Chapter 10](../10-rollouts/), and not in a good way.
