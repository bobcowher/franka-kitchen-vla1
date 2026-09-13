---
title: "What you're building"
part: "Part I · Orientation"
chapter: 1
weight: 1
standfirst: "A frozen language model, a 21,129-parameter head, and a robot arm that opens cabinets."
---

A <em class="term">VLA</em> — vision-language-action model — is a vision-language
model that outputs actions instead of words. That definition hides the only
question that matters: where does the action attach?

Pick that joint and everything follows from it. What you freeze, what you can
add later, what your loss can mean, what you can debug.

You are going to build one by hand. Not fine-tune SmolVLA, not adapt an
existing VLA through lerobot. You take a pretrained vision-language model,
freeze all of it, and write the part that turns its output into joint
velocity.

## The finished shape

SmolVLM2-500M reads one 448-pixel camera frame and one English instruction.
Every one of its 507,482,304 parameters stays frozen. An <em class="term">action
head</em> — the small trained network you write — reads two positions out of
the model's output and emits nine numbers: seven arm joint velocities and two
gripper commands.

That head is 21,129 parameters. Two `LayerNorm`s and one `Linear`. It is the
entire trained component of the system.

<div class="tokens">
  <span class="tok-img">64 image tokens</span>
  <span class="tok-txt">instruction</span>
  <span class="tok-read">read here</span>
</div>
<p class="caption">One frame plus one sentence is 79–84 tokens. Attention is causal, so each position sees only what precedes it.</p>

## Why a frozen model can work

Your instinct should be that a frozen model cannot contain robot control, and
that instinct is correct. It contains none.

What it contains is a <em class="term">representation</em>: a description of
the scene and the instruction, in 960 numbers, that a pretrained model already
computes well. The bet is that mapping "good description of a kitchen and a
goal" to "joint velocity" is simple enough for one linear layer.

That bet is cheap to test, and [Chapter 12]({{< relref "chapters/12-the-fast-loop" >}}) tests it
directly. It holds, up to a ceiling you can measure.

## What you need

The training method is <em class="term">behavior cloning</em>: you have
recordings of a human doing the task, and you train the policy to output the
action the human took, given the same observation. No reward, no exploration.
Supervised learning with a robot on the end.

<dl class="stats">
  <div><dt>Env</dt><dd>Franka Kitchen</dd></div>
  <div><dt>Demos</dt><dd>56,005</dd></div>
  <div><dt>Tasks</dt><dd>7</dd></div>
  <div><dt>GPU</dt><dd>~3 GB</dd></div>
</dl>

You need `gymnasium`, `gymnasium-robotics` for the Franka Kitchen environment,
`transformers`, and `torch` with CUDA. The VLM runs in bfloat16 and needs about
3 GB of VRAM at batch 64, so a modest card is enough. You also need
demonstration data, which this guide assumes you have — collecting it is a
separate problem involving a gamepad and some patience.

## How this guide runs

It follows the build in the order you would write it. Each chapter states the
problem, explains the reasoning, then gives you the code to write. Where a
measurement contradicted something reasonable, you get the reasonable thing
first, because otherwise the correction reads as trivia rather than a lesson.

Files you will create or modify, in order:

| chapter | file | what goes in it |
|---|---|---|
| 3 | `gym_robotics_custom.py` | environment wrappers |
| 4 | `dataset.py` | demonstration loading |
| 5–9 | `model.py` | the frozen VLM and the head |
| 10 | `agent.py` | the training loop |
| 11 | `scripts/overfit.py` | the gate |
| 12 | `scripts/probe.py` | the fast experiment loop |
| 13 | `scripts/evaluate.py` | rollout measurement |

<div class="finding">
<span class="note-label">Result</span>
The finished policy reaches <strong>96%</strong> success on one task and 68% on
another, measured over 50 rollouts each. It also cannot touch three of the
seven tasks at all. Both halves of that sentence are findings, and the second
took far longer to see than it should have.
</div>
