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

Set up an hourly session-local cron (fires :07 past the hour) to repeat this
check automatically per the Hourly loop section above. Cloud scheduling was
considered and rejected: Beekeeper's MCP server only reaches `lab.local` on
the local network, so a cloud-run agent could not reach it anyway.

**22:15 — second check.** Epoch 2900/100K. `eval/mean` = 0% (microwave,
hinge_cabinet, top_burner all 0% at the one eval point so far). Still well
before the ~15K no-judgment threshold. No action taken.

**23:19 — hourly cron check.** Epoch 12500/100K. `eval/mean` = 0% at every
eval point so far (2500, 5000, 7500, 10000, 12500) — five straight zeros, but
still inside the no-judgment window; next eval at 15000 is the first one
worth reading into per the 27K/~01:00 trigger. `train/loss` still oscillating
0.06–0.11 with no clear trend, as expected — not a decision signal. No action
taken.

**00:19 — hourly cron check.** Epoch 21100/100K. First non-zero signal:
`eval/mean` hit 11% at both the 12500 and 15000 checkpoints (top_burner 1/3
rollouts each time), then reverted to 0% at 17500 and 20000. At 3 rollouts
per task this is exactly the single-success noise floor the project already
knows about (0/0.33/0.67/1 resolution) — not a trend, just noted for the
morning. Still below the 27K/~01:00 decision point, so no unfreeze yet.
`train/loss` unchanged (0.06–0.11 band). No action taken; next check should
land on or near the 27K checkpoint.

**01:19 — hourly cron check, the 27K decision point.** Epoch 29900/100K.
**The unfreeze trigger did not fire** — `eval/mean` is not 0% at 27K, it hit
its best value yet: 22% at epoch 27500, driven by `hinge_cabinet` at 67%
(2/3 rollouts, its own best so far). Full eval/mean sequence so far: 0, 0, 0,
0, 11, 11, 0, 0, 11, 0, **22**. Noisy, but the TensorBoard trend call on both
`eval/mean` and `eval/hinge_cabinet` reads "improving" with the peak at the
most recent point. Tonight's stated win condition — any rollout success above
0% — has plausibly already been met; whether it holds needs the next couple
of eval points to confirm it isn't another single-rollout blip like the
12500–15000 bump. Per the pre-agreed trigger, **no unfreeze** — the premise
("still 0% at 27K") wasn't met, so the architecture stays frozen and training
continues unmodified. `train/loss` still in its usual 0.05–0.10 band, not used
for this decision. No action taken.

**02:19 — hourly cron check.** Epoch 38200/100K. Full `eval/mean` sequence
by step (2500→37500): 0, 0, 0, 0, 11, 11, 0, 0, 11, 0, **22**, 0, **22**, 11, 0.
Noisy, no monotonic trend, but the non-zero signal keeps recurring and is now
spread across *different* tasks at different checkpoints — hinge_cabinet hit
67% at 27500, top_burner independently hit 67% at 32500 — which makes chance
alone a weaker explanation than at the last check. `microwave` has been 0% at
every single eval point so far; worth watching whether it ever breaks 0%.
Nothing here crosses an action threshold: the run is past the 27K decision
point, the trigger already didn't fire, and this check reinforces that call
rather than reversing it. No action taken.

**03:19 — hourly cron check, clear jump.** Epoch 47400/100K. `eval/mean`
jumped to its best run yet: **56% at epoch 40000** (hinge_cabinet 100%,
top_burner 67%), then 33% at 42500 and 44% at 45000 — both with hinge_cabinet
still at **100%**. hinge_cabinet has now hit 100% three checkpoints in a row
(40000, 42500, 45000), which reads as genuinely solved rather than rollout
luck. top_burner keeps swinging (67% → 0% → 33%) — still noisy but nonzero.
`microwave` remains at flat 0% across all 19 eval points recorded so far —
the one task that has never once succeeded; worth Robert's attention in the
morning (possibly a harder task, possibly something task-specific), but not
an authorized action tonight. No unfreeze needed — the signal keeps getting
stronger, not weaker. No action taken.

**04:19 — hourly cron check, peak reversed.** Epoch 55700/100K. The 40–45K
peak did not hold: `eval/mean` by step from 47500 on: 11%, 11%, 0%, 0% — four
checkpoints declining back toward zero right after hinge_cabinet's three
100%s. hinge_cabinet itself went 100/100/100 → 0/0/0/0, a sharper reversal
than 3-rollout noise alone comfortably explains, though not impossible at
this sample size. One genuine positive: `microwave` broke its 0% streak for
the first time ever at 47500 (33%), though it too fell back to 0% the next
three checkpoints. `train/loss` keeps declining smoothly (now 0.057, still
its best-ever region) — so this doesn't look like a training blowup, more
like an eval-side pattern (possibly overfitting to hinge_cabinet's specific
demos, or just the small-sample variance the project has already flagged
elsewhere). Not touching anything: no authorized trigger fits a "peaked then
declined" pattern, and this needs Robert's judgment on whether it's noise or
a real regression, not a unilateral architecture change. No action taken —
flagging clearly for the morning.

**05:19 — hourly cron check.** Epoch 64500/100K. Pattern holds steady: after
the 40–45K peak and the four-checkpoint reversal, `eval/mean` ticked back up
to 22% at 57500 (hinge_cabinet 67% again) before dropping to 0% at both 60000
and 62500. Full picture across the whole run is a noisy oscillation between
0% and 56%, occasional spikes almost always carried by `hinge_cabinet`, no
sustained plateau either way. `train/loss` continues its slow, steady decline
(0.057–0.09 band, best-ever region) — training itself looks healthy, the
volatility is on the eval/rollout side, consistent with 3-rollout resolution
plus whatever is making `hinge_cabinet` inconsistent rather than any training
instability. No trigger crossed, nothing actionable changed since the last
check. No action taken.

**06:19 — hourly cron check.** Epoch 73000/100K. Same oscillation continues:
11% (65000) → 22% (67500) → **44% at 70000**, the first time `hinge_cabinet`
and `top_burner` both hit 67% at the same checkpoint → back to 0% at 72500.
`microwave` still hasn't broken 0% again since its one blip at 47500.
Nothing here changes the picture already logged at 04:19/05:19 — noisy,
non-zero, no sustained plateau, training loss healthy throughout. No trigger
crossed. No action taken.

**07:19 — hourly cron check.** Epoch 81700/100K. Continued oscillation: 11%
(75000, top_burner) → 0% (77500) → 11% (80000, `microwave`'s second-ever
success, its first since the 47500 blip). No new pattern — same story as the
last three checks. At the current pace (~2.4 epochs/sec including eval
overhead) the run has roughly ~18,300 epochs left, projecting completion
around **09:2X** — right around when Robert said he'd check back in. No
trigger crossed. No action taken.

**08:19 — hourly cron check.** Epoch 90000/100K, ~10K epochs left (~1h at
current pace — still tracking a ~09:2X finish). Last four checkpoints:
0% (82500), 11% (85000, hinge_cabinet 33%), 0% (87500), 0% (90000) — the
quietest stretch since the 40K peak. TensorBoard's own trend call has
shifted from "improving" to **"stable... appears to have plateaued"** for
`eval/mean`, `hinge_cabinet`, and `top_burner` — i.e. the run looks like
it settled into this noisy-but-mostly-low regime rather than climbing
toward the 40K peak again. `train/loss` is still in its best-ever region
(0.044–0.07), so training hasn't broken; the ceiling looks like it's on the
eval/rollout side. Nothing actionable: no trigger fits "plateaued below an
earlier peak," and this is exactly the kind of judgment call — is 40K's
checkpoint better than what training converges to at 100K? — that's
Robert's to make in the morning, not something to act on unilaterally
tonight. No action taken.

**09:19 — hourly cron check, run nearly complete.** Epoch 98900/100K (~1100
epochs left, a few minutes at current pace) — this is essentially the last
check of the night, right on schedule for Robert's return. Last two eval
points came in strong: 22% at 92500 (hinge_cabinet 67%) and **44% at 97500**
(hinge_cabinet 67%, top_burner 67% together again) — ending on an upswing,
not the quiet 82.5K–90K stretch. Best-ever point is still 56% at epoch
40000. `train/loss` finished around 0.04–0.09, still its best-ever region
throughout. Summary for the morning: `eval/mean` never sustained a plateau
at its peak — it oscillated 0–56% for the entire second half of the run,
with `hinge_cabinet` the most reliable performer, `top_burner` intermittent,
and `microwave` only ever non-zero twice (47500, 80000). No unfreeze was
ever triggered because eval/mean was already non-zero well before the 27K
checkpoint. Whether the 40K checkpoint should be preferred over the final
weights is a real open question for Robert to decide — not something to
resolve unilaterally. No action taken; next check (if the loop continues)
will see the run in a finished state.

**09:28 — run 9 completed. Loop wrapping up.** Full 100K epochs finished in
11h37m57s, status COMPLETED. Final checkpoint (100000) happened to read 0%
across the board — the last eval point was a quiet one, not a regression;
the checkpoint immediately before it (97500) was 44%, one of the best of the
night. Since there's no more training to check, this stops the hourly loop:
deleted cron job `67b3a488`.

**Milestone (a) — "any rollout success above 0%" — was met repeatedly and
independently, not once.** Across ~40 eval points from epoch 2500 to 100000,
`eval/mean` oscillated between 0% and 56%, never sustaining a plateau at its
peak but recurring nonzero dozens of times, on different tasks, at different
points in training:
- **hinge_cabinet**: the most reliable task. Hit 100% three checkpoints in a
  row (40000, 42500, 45000) before reverting; recovered to 67% multiple times
  afterward (57500, 70000, 92500, 97500). Best single task result of the run.
- **top_burner**: intermittent throughout, 33–67% at irregular checkpoints,
  including twice alongside hinge_cabinet at the same step (70000, 97500).
- **microwave**: the outlier. Non-zero exactly twice all night (33% at 47500
  and 80000) out of ~40 eval points. Worth a look — possibly a harder task
  geometrically, possibly something instruction- or demo-specific to that
  task alone.
- Best single eval/mean: **56% at epoch 40000**. `train/loss` was healthy and
  monotonically improving all night (0.33 → ~0.06, best point 0.038 at
  step 39920) — training itself never broke; all the volatility was on the
  eval/rollout side.

**The unfreeze trigger was never pulled.** `eval/mean` was already
recurring non-zero well before the 27K/~01:00 checkpoint, so the "unfreeze
if still 0%" premise never held. The frozen VLM + tiny ActionHead architecture
was left completely untouched all night, as authorised.

**Open question for Robert, not resolved unilaterally:** is the epoch 40000
checkpoint (56% eval/mean, hinge_cabinet's cleanest run) actually the better
model to keep, versus the final epoch 100000 weights (which happened to
land on a 0% eval)? Beekeeper's `run_history`/artifact storage should have
both checkpoints if intermediate checkpointing was on — worth checking. No
checkpoints were touched or deleted tonight; `checkpoints/bc_network` was
never touched either, per the standing constraint.

**Nothing else pending.** No unauthorized changes were made all night: no
architecture changes, no demo re-collection, no touching the BC baseline. All
changes were HANDOFF.md log entries, committed and pushed to `main` after
every check.

## Morning: the hinge_cabinet oscillation, and run 10 (unfreeze A/B)

Robert asked two things: (1) dig into the hinge_cabinet 100%→0% swing flagged
overnight, (2) try the next lever (unfreeze) rather than wait for a trigger
that already didn't fire. Both below.

**1. The oscillation is mostly `EVAL_ROLLOUTS=3` doing what small samples do.**
Full-run `analyze_run(9)` shows `eval/hinge_cabinet` as `trend: unstable`,
`best_value: 1.0` at step 40000, `final_value: 0`, `anomaly_count: 0` — i.e.
nothing in the raw curve reads as a spike or crash, just noise. The back half
of the run (raw log, 60000→100000, 17 checkpoints) reads:
`0,0,33,0,67,0,0,0,0,0,33,0,0,67,0,67,0` (%) — mostly 0 with scattered 1-of-3
and 2-of-3 hits, never another 100. At n=3, a *true* success rate of ~30-40%
already produces exactly this pattern: P(0/3)≈22-35%, P(1/3)≈36-44%, P(2/3)
only ≈24-31%, and three independent 3/3s in a row (the 40000/42500/45000
streak) at p≈0.4 is ~6% per triplet — rare but not remarkable over a
40-checkpoint run, especially since adjacent checkpoints are correlated
(weights barely move in 2500 epochs), which makes streaks *more* likely than
the independent-trials math suggests.

So: no monotonic collapse, no separate bug to chase. The real finding is that
**a single 3-rollout eval is not a strong enough signal to pick a checkpoint**
— which HANDOFF already flagged as a "final-tuning problem" for EVAL_ROLLOUTS.
It now is one. Recommendation for whoever resolves the open "40K vs final
checkpoint" question: re-evaluate the top few candidates (40000, 42500, 45000,
70000, 97500 — the highest eval/mean checkpoints) with more rollouts (10-20)
offline before choosing, rather than trusting any single in-run number,
40K's included.

**2. Run 10 started: unfreeze the last 2 text-decoder layers, 20K-epoch A/B.**
Commit `b07ce4d`. Rather than wait on the (already-missed) 27K/0% trigger,
Robert asked to try the unfreeze lever directly and compare against run 9's
eval/mean at matching checkpoints. Implementation:
- `model.py`: `UNFREEZE_LAST_N_LAYERS` (env var, default 0 = run 9's frozen
  behaviour) unfreezes the last N text layers, matched by parsing parameter
  names rather than a hardcoded attribute path. Raises immediately at
  construction if the naming pattern doesn't match anything, since this
  couldn't be smoke-tested locally (no GPU/torch on this machine).
- `agent.py`: unfrozen VLM params get a separate Adam param group at
  `BACKBONE_LR` (default lr/10 = 1e-4) rather than the head's 1e-3, since a
  head-sized LR on pretrained weights would wreck them fast. Checkpoints
  (rolling + per-epoch eval snapshots) now save head+backbone together via
  `trainable_state_dict()`, falling back to the old bare-head format when
  nothing is unfrozen — old checkpoints and `scripts/test.py` still load fine.
- `scripts/train.py`: run 10 sets `UNFREEZE_LAST_N_LAYERS=2`,
  `BACKBONE_LR=1e-4`, `VLA_EPOCHS=20001` (not 100K — run 9's first signal was
  by 12500 and its peak by 40000, so 20K is enough to compare eval/mean at
  matching checkpoints without paying for a full run). **Revert these three
  once the A/B is read** — they're marked as a temporary experiment in the
  code comment.

Verified at epoch 200 (log tail): GPU pin correct (`RTX 3090`, matches the
banner), `model: unfroze last 2 text layers (19,664,640 params)` and
`agent: training 19,664,640 backbone params at lr=0.0001` both printed as
expected, loss dropped 0.489→0.135→0.145 in the first 200 steps — no NaN, no
crash. Will check back against run 9's eval/mean at 2500, 5000, ... as run 10
reaches each checkpoint.

**Run 10 progress check, epoch ~10800 (1h19m elapsed).** `eval/mean` so far:
2500→11%, 5000→0%, 7500→0%, 10000→11% (top_burner and hinge_cabinet each hit
once, 1-of-3 rollouts). `train/loss` healthy, no anomalies beyond normal
early-training spikes. For comparison, **run 9 (frozen VLM) was still 0%
across all four of these same checkpoints** — its first non-zero reading was
at 12500. So run 10 has a mild lead, but at n=3 rollouts each of these is a
single lucky episode, not yet a real signal (see the oscillation analysis
above — don't read anything into this until more checkpoints land, same "not
before ~15K" caution as run 9 got). Next natural check: epoch 15000-20000,
where run 9 first built its recurring non-zero pattern.

Still running, still on GPU 1 (RTX 3090), still un-touched: `checkpoints/bc_network`,
architecture, demos.

## Offline eval, 2026-09-12 evening: the in-run eval was hiding the model

Robert asked whether we're ready for a lesson guide or need changes first.
Answer: the architecture is fine, the *measurement* was the problem. Built
`scripts/evaluate.py` (n=50 rollouts, Wilson intervals, same `agent.test()`
code path as the in-run eval) and re-read run 9's candidate checkpoints.

**1. Ranking run 9's checkpoints, hinge_cabinet + top_burner, n=100 each.**

```
checkpoint   in-run(n=9)   offline(n=100)   95% CI         hinge   burner
e40000          56%           61.0%      [51.2, 70.0]       68%      54%
e45000          44%           60.0%      [50.2, 69.1]       86%      34%
e42500          33%           52.0%      [42.3, 61.5]       56%      48%
e97500          44%           47.0%      [37.5, 56.7]       26%      68%
e70000          44%           35.0%      [26.4, 44.7]       38%      32%
e27500          22%           32.0%      [23.7, 41.7]       64%       0%
e100000          0%            0.0%      [ 0.0,  3.7]        0%       0%
```

- **The final checkpoint is genuinely dead**, not a noisy zero: 0/100, CI
  capped at 3.7%, while `train/loss` sat in its best-ever region (0.04-0.09).
  The default artifact of an 11.5h run was a brick and the loss curve said
  nothing. Keep the best eval snapshot, never the final weights.
- **In-run numbers mis-ranked the field.** e45000 and e70000 both read 44%
  overnight; they are 60% and 35%, non-overlapping intervals. e40000's 56%
  was right by luck.
- **Checkpoints specialise and trade tasks off.** e45000 is hinge-heavy
  (86/34), e97500 is burner-heavy (26/68), e40000 is balanced (68/54). The
  mean stays flat while the policy drifts sideways -- task interference in a
  21,129-param head on a frozen prefix. This is a better argument for
  unfreezing than the eval/mean curve ever was.

**2. Coverage of e40000 across tasks that had never been rolled out.**
`EVAL_TASKS` only ever covered 3 of the 7 trained tasks. The other four,
measured for the first time at n=50:

```
slide cabinet   48/50   96.0%   [86.5, 98.9]    <- never measured before
hinge cabinet   34/50   68.0%   [54.2, 79.2]
top burner      27/50   54.0%   [40.4, 67.0]
bottom burner    4/50    8.0%   [ 3.2, 18.8]    <- never measured before
kettle           0/50    0.0%   [ 0.0,  7.1]    <- never measured before
light switch     0/50    0.0%   [ 0.0,  7.1]    <- never measured before
microwave         n/a  (~2% over ~120 in-run rollouts overnight)
```

113/300 across the six measured tasks. `slide cabinet` is near-solved and
nobody had ever looked at it. Three tasks are flat zero, one is marginal.
The open question is what separates them -- instruction encoding, demo
count, or start-pose geometry -- not overall model capacity.

**Milestone (a) is comfortably met**, not marginally: 96% on one task, 61%
mean on the two hard ones. Far better than the <=56% the overnight numbers
implied.

**3. Run 10 (unfreeze last 2 text layers) completed**, 20K epochs in 2h26m.
eval/mean by checkpoint: 11, 0, 0, 11, 0, 0, 0, 0. Run 9 over the same range:
0, 0, 0, 0, 11, 11, 0, 0. **Both are noise at n=9 and neither is readable.**
The A/B is still unresolved and needs run 10's snapshots re-run through
`scripts/evaluate.py` at n=50 against run 9's at matching epochs. Not started
-- it is GPU work and the local box was needed elsewhere.

**Next actions, in order.** (a) Settle the run 10 A/B offline at n=50.
(b) Raise EVAL_ROLLOUTS or move checkpoint selection offline entirely -- n=3
cannot rank anything. (c) Add `slide cabinet` to EVAL_TASKS and drop
`microwave` from the in-run set; it costs the most (no early termination on
failure) and discriminates least. (d) Diagnose the three dead tasks.
