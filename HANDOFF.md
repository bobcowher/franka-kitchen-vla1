# Handoff — 2026-09-11, 22:00

Overnight context for an agent picking this up cold. Robert is asleep.

## What this project is

Build a VLA by adding action capability to a small VLM **by hand**. Not SmolVLA,
not lerobot, not fine-tuning someone else's VLA. Learning the construction is the
point, so the architecture is deliberately the smallest honest one.

The policy: frozen SmolVLM2-500M reads one 448px frame and one English
instruction. An `ActionHead` — two LayerNorms and a `Linear(1920 → 9)`, 21,129
parameters — reads two positions out of the prefix and emits joint velocity.

Current milestone is **(a): the arm moves sensibly on one task.** Not beating the
BC baseline. Any rollout success above 0% is the win condition tonight.

## Right now

**Run 9 is training.** Beekeeper project `franka-kitchen-vla1`, commit `e9a8b10`,
100K epochs, batch 64, on lab's RTX 3090.

Measured, so you can budget:

| | |
|---|---|
| throughput | 2.55 epochs/sec |
| 100K epochs | ~10.9 h + ~100 min of evals ≈ 12.6 h |
| eval cadence | every 2500 epochs, 3 rollouts × 3 tasks |
| epoch 27K reached | ~01:00 |
| VRAM / RAM | 2.9 GB / 36.2 GB |

**Primary metric: `eval/mean`, higher is better.** Per-task rates can only be 0,
0.33, 0.67 or 1 at three rollouts, so the mean across tasks is the only eval
number with resolution. **Never tune on `train/loss`** — it is anti-correlated
with success in this project's history.

## The fast loop — use this, not training runs

`python scripts/probe.py` is the most valuable tool here. The VLM is frozen, so
its output for a given (frame, instruction) never changes: encode a sample once
(~60 s), then fit any readout or head idea on the cached vectors in seconds.
Held out by shard, 5 seeds, because one seed swings by more than the effects
being measured.

Current table, held-out weighted MSE, predict-the-mean baseline 0.2132:

```
head                            params     mean     best    worst
last only                       10,569   0.1090   0.1025   0.1148
pooled only                     10,569   0.0972   0.0930   0.1030
fused (shipped)                 21,129   0.0905   0.0885   0.0934
fused -> 512 -> 9              992,009   0.1519   0.1057   0.2133
```

Anything that changes only the head, the readout, the loss weighting or the
normalisation should be answered here first. It costs minutes.

## Hourly loop

Robert asked for hourly iteration. Each hour:

1. `analyze_run("franka-kitchen-vla1")` and `get_logs(tail=40)`.
2. Read `eval/mean`. Compare to the last check.
3. Decide, act, and **append a dated entry to this file** saying what you did and
   why. The morning handoff is this document.

Trigger already agreed with Robert: **if `eval/mean` is still 0% at ~27K
(around 01:00), the frozen prefix is the suspect and unfreezing is the next
lever** — LoRA, or the last few text layers. That premise change is authorised
(see below). Note it invalidates the probe cache, since the prefix stops being
fixed.

Do not read anything into 0% before ~15K. The BC conv stack did not sweep until
27K, and 3 rollouts in that regime is where this project has previously built
theories on noise.

## Authorised

Robert, verbatim: *"full access to do what it needs to do without completely
ripping out the architecture. It should tweak hourly through the night."*

**Yes:**
- Stop and start Beekeeper runs, change hyperparameters, change the head or
  readout, add the `joint_pos` state token, try action chunking.
- Unfreeze the VLM (LoRA or last-N layers) if the 27K check reads 0%.
- Commit and push to `main`. Keep commits explaining *why*, with measurements.

**No:**
- Replacing the architecture. No SmolVLA, no lerobot dataset format or training
  loop, no going back to a conv stack. The hand-built VLM-plus-head is the point
  of the project.
- Touching `checkpoints/bc_network` — that is the 44.9 MB BC baseline from
  run 480 and the only copy. The VLA writes `vla_network`.
- Re-collecting demos. That costs Robert joystick time.

## Decisions already made — do not relitigate

Each of these was argued and settled today. Reasons matter more than the answers.

- **Live forward pass, not a precomputed prefix.** Caching the VLM output in the
  *training loop* forecloses putting action queries in the sequence, which
  chunking needs. (Caching for `probe.py` is fine — that is a measurement, not
  the architecture.)
- **Continuous output, no discretised action tokens.** Robert's call; the field
  has moved away from binning.
- **k=1 before chunking.** Chunking is wanted eventually, but k=1 is the version
  where overfitting ten samples to zero loss is unambiguous.
- **`EVAL_ROLLOUTS = 3` stays.** Robert's correction: 3 cannot separate two
  near-saturated configs, but that is a final-tuning problem. Day 1, any success
  over 0 is the signal and cheap evals buy iteration speed.
- **No action normalisation.** Actions already *are* joint velocity in [−1, 1]
  (`kitchen_env.py:89`). Adding quantile scaling would normalise twice.
- **No `tanh` on the output.** With ±1 gripper targets it makes zero loss
  unreachable, which blunts `scripts/overfit.py`, the only unambiguous gate here.
  The env clips to the velocity bounds anyway.
- **`joint_pos` is deliberately absent.** Worth +14.7pp per the BC ablation, but
  with proprioception wired to the head the model can partially fit while the
  prefix is garbage, which blinds the gate. It goes in as a `Linear(9 → 960)`
  prefix token when chunking arrives.

## Traps, each of which has already cost time

- **`CUDA_VISIBLE_DEVICES` means a different card than `nvidia-smi` says.** CUDA
  orders devices by speed, `nvidia-smi` by PCI bus, and on lab they are inverted.
  `scripts/train.py` pins `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Beekeeper does not, so
  **its launch banner reports a GPU the process is not on.** Four runs went to
  the 3060 while the banner said 3090. Verify with
  `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory` and map the UUID —
  never trust the banner or the index. Full writeup in
  `~/pythonprojects/beekeeper/bug-gpu-device-order.md`.
- **Cosine similarity cannot see anything here.** SmolLM2's residual stream has
  massive activation outliers — one dim carries 60× the median magnitude — so any
  two hidden states read ~0.99 regardless. Decompose `h = mean + deviation` and
  report the ratio instead.
- **One seed is not a measurement.** The fused head read 0.0934 and 0.1197 on
  identical data. `probe.py` runs 5 seeds for this reason. Do not report a
  single-seed number as a result.
- **Frames are HWC on both sides of the train/eval line.** They disagreed for one
  commit and the rollout died on a channel count. The silent version of that bug
  trains on transposed pixels and is merely worse.
- **Run the rollout path before committing to a long run.** It is ~8 seconds and
  it is the only thing that exercises the eval half of the code.
- **Prompts are 79–84 tokens and right-padded.** The head gathers each row's last
  *real* token at `attention_mask.sum(1) - 1`. Reading `[:, -1]` would read a pad
  for five of seven tasks, silently. Left-padding would shift every RoPE position.

## What is known about the ceiling

Measured today and the most important thing to know before spending the night on
head tweaks: **head capacity is not the bottleneck.** At 10,569 params train sits
at 0.083 against held-out 0.102, and adding 500K or 1.5M parameters makes both
worse. Every readout variant plateaus around 0.09 against a 0.213 baseline — so
roughly 56% of variance explained, and the frozen prefix is what caps it.

For scale: the BC conv stack reached 100% success at train loss ~2e-4. We are two
orders of magnitude above that. Whether 0.09 converts to *any* rollout success is
genuinely unknown, and is the question run 9 answers.

If it does not, the levers in order of expected value are: unfreeze (LoRA), then
the `joint_pos` state token, then chunking. Not a bigger head.

## Leave for the morning

- Anything that needs Robert's judgment on direction rather than measurement.
- The lesson guide at
  https://claude.ai/code/artifact/32ce4858-6bf2-4f03-8b4a-9e21072e4679 — it
  documents the build and is now behind on the fused readout and the seed-variance
  finding. Worth updating, but it is a deliverable for him, not an overnight task.

## Orientation commands

```
get_project_instructions("franka-kitchen-vla1")
training_status("franka-kitchen-vla1")
analyze_run("franka-kitchen-vla1")
get_logs("franka-kitchen-vla1", tail=40)
```

Dataset on lab: `/data/datasets/farama-kitchen-bc/dataset` — note *farama*, where
the local symlink says *franka*. 582 shards, 56,005 steps, 896px archive.
`DATASET_PATH` carries it; do not hardcode.

---

## Log

**22:00 — handoff written.** Run 9 at ~1K epochs, loss 0.099, no eval yet.

**22:08 — first check-in via Beekeeper MCP.** Run 9 at epoch 2500/100K (~18 min
elapsed). GPU pin confirmed working: launch banner shows GPU 1 = the 3090, matching
`e9a8b10`. `train/loss` oscillating 0.07–0.11, no trend to act on. First eval:
microwave 0%, hinge cabinet 0% — expected this early, well before the ~15K
no-judgment threshold. No action taken.
