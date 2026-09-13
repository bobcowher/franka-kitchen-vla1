---
title: "The dataset"
part: "Part II · The pieces"
chapter: 4
weight: 4
standfirst: "56,005 demonstration steps, why they live in RAM, and two things the data tells you before training starts."
---

Demonstrations arrive as 582 `.npz` shards, one per recorded episode. Each
holds a sequence of frames, joint states, actions and a task label.

A <em class="term">shard</em> is one continuous episode: a human driving the
arm through one task from reset to success. This matters later, because
consecutive steps inside a shard are nearly identical — the arm has moved a
few millimetres — and that fact breaks naive validation splits.

## Preallocate, don't append

<p class="filename">Filename: <strong>dataset.py</strong></p>

```python
class Dataset():
    def __init__(self, max_size, image_size, n_actions, n_joints=9):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.camera_scene_memory = np.zeros(
            (self.mem_size, image_size, image_size, 3), dtype=np.uint8)
        self.joint_pos_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.joint_vel_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.task_id_memory = np.zeros(self.mem_size, dtype=np.int64)
```

One contiguous array per field, sized up front. Sampling a batch is then one
fancy-index operation with no copying of Python objects.

**The image size is the lever that decides whether this fits in memory.** A
step costs 2.30 MiB at 896 pixels, 588 KiB at 448, and 147 KiB at 224. At
56,005 steps, 448 costs about 36 GB and 896 would cost 129 GB. The policy
trains at 448 for that reason and no other.

`load_data` counts the steps in every shard before writing anything, and raises
if the total exceeds `max_size`. An oversized directory should be an error, not
a machine that starts swapping.

## Sampling

<p class="filename">Filename: <strong>dataset.py</strong></p>

```python
def sample_batch(self, batch_size):
    batch = np.random.choice(self.mem_ctr, batch_size)

    state = {
        # Left HWC. The conv stack wanted NCHW; Model.preprocess permutes on
        # the GPU instead, so transposing here would only be undone there.
        "camera_scene": self.camera_scene_memory[batch],
        "joint_pos": self.joint_pos_memory[batch],
        "joint_vel": self.joint_vel_memory[batch],
    }

    return (state,
            self.action_memory[batch],
            self.reward_memory[batch],
            self.terminal_memory[batch],
            self.task_id_memory[batch])
```

Frames stay in HWC order, matching `ObsReshapeWrapper` from the previous
chapter. That comment is in the source for a reason, and
[Chapter 14]({{< relref "chapters/14-traps" >}}) is the reason.

## Two things the data tells you

Before training, run statistics over all 56,005 actions. Two findings change
the code.

**The gripper is binary.** Dimensions 7 and 8 are exactly ±1 in 100% of steps,
never anything between, with a standard deviation of 0.902 against roughly 0.29
for a typical arm joint. Left alone they carry about 8× an arm joint's variance
and dominate the gradient, so the model optimizes an easy binary decision at
the expense of the seven continuous ones that position the arm. Chapter 10
weights them down.

**Dimension 4 is dead.** Standard deviation 0.017, with the 1st and 99th
percentiles both exactly 0.000 — zero in over 98% of steps. Nothing on the
gamepad used for collection maps to that joint.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
Your demonstrations are effectively 6 arm degrees of freedom plus a gripper,
not 7. Know that before you draw any conclusion about which poses the policy
can reach. Statistics over your own action data is twenty lines and it changes
the loss function.
</div>
