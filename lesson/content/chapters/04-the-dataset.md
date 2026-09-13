---
title: "The dataset"
part: "Part II · The pieces"
chapter: 4
weight: 4
standfirst: "56,005 demonstration steps, why they all live in RAM, and two facts that change the loss function."
---

Like the environment, `dataset.py` came over from the behavior-cloning project,
and you'll change one line of it. The listings show the file after that change,
and the edit itself is spelled out right after Listing 4.2. We're walking through
the rest because the batch
it produces is what the model consumes, and because there are two things hiding
in this data that will change code we write in Chapter 10.

## What a shard contains

Demonstrations arrive as 582 `.npz` files, one per recorded episode. Let's open
one:

```python
path = sorted(glob.glob("dataset/microwave/*.npz"))[0]
d = np.load(path)
print("keys:", list(d.keys()))
for k in ("camera_scene", "action", "joint_pos"):
    print(f"  {k:<14} {d[k].shape}  {d[k].dtype}")
print("task_description:", repr(str(d["task_description"])))
```

<div class="output"><p class="output-label">This prints</p>

```text
keys: ['task_name', 'task_description', 'camera_scene', 'joint_pos',
       'joint_vel', 'action', 'reward', 'done']
  camera_scene   (75, 896, 896, 3)  uint8
  action         (75, 9)  float64
  joint_pos      (75, 9)  float32
task_description: 'Open the microwave door'
```
</div>

Seventy-five steps of a person opening a microwave, with frames stored at full
896-pixel resolution. We'll call one of these files a *shard*, and it's worth
noticing now that consecutive steps within one are nearly identical pictures,
because that fact breaks the obvious way of splitting data for validation. We'll
come back to it in Chapter 12.

## Preallocate, don't append

<p class="listing">Listing 4.1 <em>One contiguous array per field, sized up front</em></p>
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

Sampling a batch then becomes one fancy-index operation per field, with no
Python-level copying.

The image size is what decides whether any of this fits in memory. A single step
costs 2.30 MiB at 896 pixels, 588 KiB at 448, and 147 KiB at 224. Across 56,005
steps that's 129 GB, 36 GB, or 9 GB respectively, and it is the only reason the
policy trains at 448 rather than at the resolution the shards were recorded in.

`load_data` counts the steps in every shard before writing anything and raises if
the total would exceed `max_size`. An oversized directory should be an error you
see immediately, not a machine that starts swapping an hour into a run.

## Sampling a batch

<p class="listing">Listing 4.2 <em>One fancy index per field</em></p>
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

That commented line is the one edit in this file, and your copy doesn't have it
yet. In yours, the line reads:

```python
"camera_scene": self.camera_scene_memory[batch].transpose(0, 3, 1, 2),
```

Delete `.transpose(0, 3, 1, 2)`, then replace the comment above the line with the
one from Listing 4.2, since the old comment describes a conv stack that no longer
exists.

That transpose produced the channels-first batch a `Conv2d` expects. Frames now
stay HWC, which matches the `ObsReshapeWrapper` you changed in the previous
chapter, and Chapter 9 does the permute on the GPU where it costs nothing. With
both edits in, the dataset and the environment agree on axis order again.

## Two facts in the actions

Before writing a loss function, it's worth looking at the numbers we're asking
the model to reproduce. Let's print per-dimension statistics over one shard:

```python
a = d["action"]
print(f"{'dim':>4}{'std':>9}{'min':>8}{'max':>8}{'|x|==1':>9}")
for i in range(a.shape[1]):
    frac = float((np.abs(a[:, i]) > 0.999).mean())
    print(f"{i:>4}{a[:, i].std():>9.3f}{a[:, i].min():>8.2f}"
          f"{a[:, i].max():>8.2f}{frac:>9.1%}")
```

<div class="output"><p class="output-label">This prints</p>

```text
 dim      std     min     max   |x|==1
   0    0.370   -1.00    1.00     1.3%
   1    0.013    0.00    0.12     0.0%
   2    0.055   -0.16    0.15     0.0%
   3    0.182   -0.57    0.35     0.0%
   4    0.000    0.00    0.00     0.0%
   5    0.340   -1.00    0.00    13.3%
   6    0.000    0.00    0.00     0.0%
   7    0.779   -1.00    1.00   100.0%
   8    0.779   -1.00    1.00   100.0%
```
</div>

Two things jump out of that table.

**The gripper is binary.** Dimensions 7 and 8 are ±1 in 100% of steps and never
anything in between, with a standard deviation of 0.779 here and 0.902 across
the whole dataset, against roughly 0.29 for a typical arm joint. Left alone they
carry about eight times an arm joint's variance and will dominate the gradient,
so the model would happily optimize an easy binary decision at the expense of
the seven continuous numbers that actually position the arm. Chapter 10 weights
them down by a factor of eight for exactly this reason.

**Dimension 4 is dead.** Its standard deviation over the full dataset is 0.017,
and the 1st and 99th percentiles are both exactly 0.000, meaning it is zero in
over 98% of all steps. Nothing on the gamepad used for collection maps to that
joint. (On this particular shard dimension 6 is flat as well, though across the
dataset it does move.)

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
Run those statistics over your own demonstrations rather than trusting ours.
Twenty lines of numpy will tell you which dimensions are binary, which are dead,
and how much they differ in scale, and all three of those change what your loss
function should look like.
</div>

<div class="exercise">
<h4>Exercise 4.1 &nbsp;What the arm can actually reach</h4>
<p>Dimension 4 being zero in 98% of steps means the demonstrations are
effectively six arm degrees of freedom plus a gripper, not seven. Pick a shard,
replay its actions into a fresh environment, and compare the final joint
positions against the recorded <code>joint_pos</code>.</p>
<p>Then ask the more interesting question: are there poses relevant to any of
the seven tasks that six degrees of freedom cannot reach? Chapter 13 finds three
tasks the policy never once succeeds at, and this is one of the candidate
explanations we never ruled out.</p>
</div>

Next, we'll start on the part that's genuinely ours to write, beginning with
loading the vision-language model and making it fourteen times cheaper to run.
