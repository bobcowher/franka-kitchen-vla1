---
title: "Where the action attaches"
part: "Part I · Deciding what to build"
chapter: 1
weight: 1
standfirst: "A VLA is a language model that emits torque instead of text. Every real decision follows from one joint."
---

A vision-language-action model is a vision-language model that outputs actions
instead of words. That sentence is accurate and almost useless, because it
hides the only question that matters: **where does the action attach?**

Pick that joint and everything else is downstream — what you freeze, what you
can chunk later, what your loss can mean, what you can debug. Pick it badly
and you find out four hours into a training run.

## The shape of the thing

Here is the whole policy, and it fits in a paragraph. SmolVLM2-500M reads one
448px camera frame and one English instruction. The model is frozen — all
507,482,304 parameters of it. An action head reads two positions out of the
resulting hidden states and emits nine numbers: seven arm joint velocities and
two gripper commands.

That head is **21,129 parameters**. Two LayerNorms and a `Linear(1920 → 9)`.
It is, genuinely, the entire action capability of the system.

<div class="tokens">
  <span class="tok-img">64 image tokens</span>
  <span class="tok-txt">instruction</span>
  <span class="tok-read">read here</span>
</div>
<p class="caption">One frame plus one sentence is 79–84 tokens. Attention is causal, left to right.</p>

## Why a frozen model can work at all

The instinct is that a frozen model cannot possibly contain robot control, and
the instinct is half right. It contains no robot control whatsoever. What it
contains is a *representation* — a description of the scene and the
instruction, in 960 dimensions, that a pretrained model already computes well.

The bet is that the mapping from "good description of a kitchen and a goal" to
"joint velocity" is simple enough for a linear layer, given a good enough
description. That bet is testable, it is cheap to test, and
[Chapter 9](../09-the-fast-loop/) tests it directly. It turns out to be
roughly true, with a ceiling we can measure.

<div class="finding">
<span class="note-label">Result</span>
It works. That 21,129-parameter head drives the arm to <strong>96%</strong>
success on one task and 68% on another, at 50 rollouts each. It also cannot
touch three of the seven tasks at all. Both halves of that sentence are
findings, and the second one took longer to see than it should have.
</div>

## What this guide assumes

That you can read PyTorch, that you know what behavior cloning is, and that
you have seen a transformer before. It does not assume you have built a
policy on top of a language model, which is the part worth writing down.

It follows the build in the order it actually happened, including the
detours. Where a measurement contradicted something reasonable, the
reasonable thing is stated first — otherwise the correction reads like
trivia instead of a lesson.
