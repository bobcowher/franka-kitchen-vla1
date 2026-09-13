---
title: "What you're building"
part: "Part I · Orientation"
chapter: 1
weight: 1
standfirst: "A frozen language model, a small head you write yourself, and a robot arm that opens cabinets."
---

A vision-language-action model, or VLA, is a vision-language model that outputs
robot actions instead of text. Most of the published ones are large, and most
of the material written about them is about fine-tuning something that already
exists. We are going to go the other way: take a small pretrained
vision-language model, freeze every weight in it, and write the piece that
turns its output into joint velocity ourselves.

That piece is small enough to read in one sitting, which is rather the point.
By the end you will have a policy that drives a simulated Franka arm around a
kitchen, and you will know why each line of it is there.

## The finished shape

The model is SmolVLM2-500M. It reads one 448-pixel camera frame and one English
instruction, something like "Open the microwave door", and returns a sequence of
hidden states, one 960-number vector per input position.

All 507,482,304 of its weights stay frozen. The only part that trains is the
<em class="term">action head</em>, which reads two of those positions and
produces nine numbers: seven arm joint velocities and two gripper commands. It
comes to 21,129 parameters, which is two `LayerNorm`s and a `Linear` layer.

<div class="tokens">
  <span class="tok-img">64 image tokens</span>
  <span class="tok-txt">instruction</span>
  <span class="tok-read">read here</span>
</div>
<p class="caption">One frame plus one sentence is 79–84 tokens. Attention is causal, so each position sees only what precedes it.</p>

## Why freezing most of it works

A frozen language model contains no robot control, and nothing we do here will
put any there. What it does contain is a description of the scene in front of
the camera and of the goal in the instruction, in a form a pretrained model
already computes well. We will call that description a
<em class="term">representation</em>, and getting a good one is most of what
the 507 million frozen parameters are for.

The bet is that going from a good description of a kitchen and a goal to a
joint velocity is a simple enough mapping for a single linear layer to learn.
It turns out to be roughly true, and Chapter 12 shows you how to measure the
point where it stops being true.

## What you need

The training method is <em class="term">behavior cloning</em>. We have
recordings of a human driving the arm through each task, and we train the
policy to output the action the human took given the same observation. There is
no reward and no exploration, so this is ordinary supervised learning with a
robot on the end of it.

<dl class="stats">
  <div><dt>Env</dt><dd>Franka Kitchen</dd></div>
  <div><dt>Demos</dt><dd>56,005</dd></div>
  <div><dt>Tasks</dt><dd>7</dd></div>
  <div><dt>GPU</dt><dd>~3 GB</dd></div>
</dl>

You will need `gymnasium` and `gymnasium-robotics` for the Franka Kitchen
environment, `transformers` for the VLM, and `torch` with CUDA. The VLM runs in
bfloat16 and uses about 3 GB of VRAM at batch 64, so a fairly modest card is
enough.

You will also need demonstration data, which this guide assumes you already
have. Collecting it involves a gamepad and a good deal of patience, and it is a
different problem from the one we are solving here.

## How this guide runs

Chapters follow the order you would write the code in. Each one sets up a
problem, works through the reasoning, gives you the code, and ends with
something specific to print so you can check that what you just wrote does what
it should.

Where a measurement contradicted something reasonable, you get the reasonable
version first. Reading the correction without the thing it corrects turns a
lesson into trivia.

| chapter | file | what goes in it |
|---|---|---|
| 3 | `gym_robotics_custom.py` | environment wrappers |
| 4 | `dataset.py` | demonstration loading |
| 5–9 | `model.py` | the frozen VLM and the head |
| 10 | `agent.py` | the training loop |
| 11 | `scripts/overfit.py` | the gate |
| 12 | `scripts/probe.py` | the fast experiment loop |
| 13 | `scripts/evaluate.py` | rollout measurement |

Parts I and II build the policy. Part III is about telling whether it works,
which took us considerably longer to get right than the model did, and Part IV
is what we learned from that.

<div class="finding">
<span class="note-label">Where this ends up</span>
The finished policy succeeds on 96% of attempts at sliding a cabinet open and
68% at opening a hinged one, measured over 50 rollouts each. It also cannot do
three of the seven tasks at all, scoring zero on every attempt. Working out why
the second sentence is true took far longer than getting the first one.
</div>
