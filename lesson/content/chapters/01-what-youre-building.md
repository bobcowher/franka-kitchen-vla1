---
title: "What you're building"
part: "Part I · Orientation"
chapter: 1
weight: 1
standfirst: "A frozen language model, a small head you write yourself, and a robot arm that opens cabinets."
---

A *vision-language-action model*, usually shortened to VLA, is a vision-language
model that outputs robot actions instead of text. Most of the published ones are
large, and most of the material written about them is about fine-tuning a model
someone else has already built. In this guide we're going to go the other way:
we'll take a small pretrained vision-language model, freeze every weight in it,
and write the piece that turns its output into joint velocity ourselves.

That piece is small enough to read in one sitting, which is rather the point. By
the end you'll have a policy that drives a simulated Franka arm around a
kitchen, and you'll know why every line of it is there.

If you've trained a convolutional policy before but never put a language model
in the loop, this is the part that will feel unfamiliar, and it's worth saying
up front that it felt unfamiliar to us too. Several of the things we tried first
were wrong in ways that took hours to notice. Those attempts are kept in the
guide rather than tidied away, because the reasons they were wrong are more
useful than the code that eventually worked.

## Where this starts

This isn't a from-scratch project, and you should know that before you plan your
afternoon. The repository began as a copy of a working
behavior-cloning setup for the same environment: a convolutional policy, the
Franka Kitchen wrappers, a demonstration collector driven by a gamepad, and a
dataset loader. All of that existed and worked before any of this started.

What we're doing is a conversion. We take that project and replace its
perception and policy with a frozen vision-language model and a head we write.
Here is the honest accounting of what that touched:

| file | what happened to it |
|---|---|
| `model.py` | rewritten completely |
| `agent.py` | substantially rewritten |
| `scripts/overfit.py` | new |
| `scripts/probe.py` | new |
| `scripts/evaluate.py` | new |
| `dataset.py` | **one line** |
| `gym_robotics_custom.py` | **two lines** |
| everything else | untouched |

Those three lines are all the same change, and Chapters 3 and 4 explain it: the
conv policy wanted its frames in channels-first order, and the VLA wants them
channels-last. One `transpose` comes out of the dataset, another comes out of
the environment wrapper, and a `Box` shape is rewritten to match.

If you're following along without a behavior-cloning project of your own, read
Chapters 3 and 4 as a description of the ground you need to be standing on
rather than as code to type. They're there because you cannot understand the
pieces that follow without knowing exactly what an observation is and where the
demonstrations come from, not because converting them is any of the work.

## The finished shape

The model is SmolVLM2-500M. It reads one 448-pixel camera frame and one English
instruction, something like "Open the microwave door", and returns a sequence of
hidden states, which is one vector of 960 numbers for every position in its
input.

Of its 507,482,304 weights we keep 460,173,504, and every one of those stays
frozen. The only part that trains is the
*action head*, a small network that reads two of those positions and produces
nine numbers: seven arm joint velocities and two gripper commands. It comes to
21,129 parameters, which in practice means two `LayerNorm`s and a `Linear`
layer.

<div class="tokens">
  <span class="tok-img">64 image tokens</span>
  <span class="tok-txt">instruction</span>
  <span class="tok-read">read here</span>
</div>
<p class="caption">One frame plus one sentence comes to 79–84 tokens. Attention is causal, so each position sees only what precedes it. Chapter 7 explains why we read the two positions we do.</p>

## Why freezing most of it works

Think of someone who can describe a kitchen in precise detail over the phone,
right down to which cupboard door is ajar and how far, but who has never picked
up a pan. That description is genuinely hard to produce, and a pretrained
vision-language model already produces it. What it has never done is move
anything.

So we leave the describing to the frozen model and add a very small pair of
hands that listens to the description and turns it into motion. In the chapters
ahead we'll call that description a *representation*, and producing a good one
is most of what those 460 million frozen parameters are for.

The bet we're making is that going from a good description of a kitchen and a
goal to a joint velocity is a simple enough mapping for a single linear layer to
learn. It turns out to be roughly true. In Chapter 12 we'll build a tool that
measures exactly where it stops being true, which is a more interesting number
than it sounds.

## What you'll need

The training method is *behavior cloning*. We have recordings of a human driving
the arm through each task, and we train the policy to output the action the
human took when it sees the same observation. There's no reward and no
exploration, so this is ordinary supervised learning with a robot on the end of
it.

<dl class="stats">
  <div><dt>Env</dt><dd>Franka Kitchen</dd></div>
  <div><dt>Demos</dt><dd>56,005</dd></div>
  <div><dt>Tasks</dt><dd>7</dd></div>
  <div><dt>GPU</dt><dd>~3 GB</dd></div>
</dl>

You'll need `gymnasium` and `gymnasium-robotics` for the Franka Kitchen
environment, `transformers` for the vision-language model, and `torch` with
CUDA. The model runs in bfloat16 and uses about 3 GB of VRAM at batch 64, so a
fairly modest card is enough; nothing here needs a datacenter.

You'll also need demonstration data, which this guide assumes you already have.
Collecting it involves a gamepad and a good deal of patience, and it's a
different problem from the one we're solving here.

## How the guide is arranged

Chapters follow the order you would work through the conversion in. Each one sets up a problem,
works through the reasoning, gives you the code, and finishes with something
specific to print so you can check that what you just wrote does what it should.
Those checks matter more than usual here, because most of the ways this build
goes wrong don't raise an exception; they just train a slightly worse policy and
say nothing.

Where a measurement contradicted something reasonable, you'll get the reasonable
version first. Reading a correction without the thing it corrects turns a lesson
into trivia.

| chapter | file | you will |
|---|---|---|
| 3 | `gym_robotics_custom.py` | read it, then change two lines |
| 4 | `dataset.py` | read it, then change one line |
| 5–9 | `model.py` | write it |
| 10 | `agent.py` | rewrite the training loop |
| 11 | `scripts/overfit.py` | write it |
| 12 | `scripts/probe.py` | write it |
| 13 | `scripts/evaluate.py` | write it |

<div class="note">
<span class="note-label">When a listing and your file disagree</span>
The <a href="https://github.com/bobcowher/franka-kitchen-vla1">repository</a> is
the ground truth, and it holds the finished state of every file. If something
here doesn't behave the way the output says it should, diff your file against
the one in the repo before debugging anything else. Two deliberate differences
to expect: Part II teaches the pre-unfreeze build, so the repo's
<code>model.py</code> and <code>agent.py</code> also carry the Chapter 15
machinery, and the repo's <code>agent.py</code> skips allocating the 36 GB frame
buffer when constructed for evaluation.
</div>

Parts I and II build the policy. Part III is about telling whether it works,
which took us a great deal longer to get right than the model did, and Part IV
collects what we learned from that.

<div class="finding">
<span class="note-label">Where this ends up</span>
The finished policy succeeds on 96% of attempts at sliding a cabinet open and
68% at opening a hinged one, measured over 50 rollouts each. It also can't do
three of the seven tasks at all, scoring zero on every attempt we've run. Working
out why that second sentence is true took considerably longer than getting the
first one.
</div>

Next, we'll look at four decisions that get made before any code is written,
because each of them closes off options later and they're much cheaper to get
right now than to reverse once something is trained.
