---
title: "Four decisions before any code"
part: "Part I · Deciding what to build"
chapter: 2
weight: 2
standfirst: "Each of these forecloses something. Getting them in the right order is most of the work."
---

Four choices were made before a line was written. None of them is obviously
correct, all of them close doors, and each one is the kind of thing that is
expensive to reverse once there is a trained checkpoint depending on it.

## Live forward pass, not a cached prefix

The VLM is frozen, so its output for a given (frame, instruction) pair never
changes. You could run it once over all 56,005 demonstration steps, cache the
hidden states, and train the head on cached vectors. Training would be
*enormously* faster — seconds per epoch instead of hundreds of milliseconds
per batch.

It was rejected, and the reason is worth being precise about: caching the
output permanently forecloses putting anything **into** the sequence. The
moment you want a state token for proprioception, or learned query tokens for
action chunking, or to unfreeze a layer, every cached vector is garbage.

> Freezing the weights saves you the backward pass. Caching the outputs saves
> you the backward pass and costs you the architecture's future.

The cache comes back in [Chapter 9](../09-the-fast-loop/), but as a
*diagnostic* tool with a known expiry, not as the training path.

## Continuous output, not action tokens

OpenVLA and its descendants discretize: bin each action dimension into 256
buckets, map the bins onto unused vocabulary entries, and let the existing
language-model head predict them. It is elegant — zero new parameters — and
it is what a lot of published work does.

Two costs made it wrong here. It needs nine sequential decode steps to emit
one action, and its loss is a cross-entropy over buckets, which never reaches
an interpretable zero. That second one matters more than it sounds, because
[Chapter 8](../08-the-gate/) is built entirely on "can this reach exactly
zero on ten samples," and a quantization floor would blunt the only
unambiguous diagnostic in the design.

The field has been moving toward continuous output regardless.

## Joint velocity, emitted directly

The reflex is to normalize the action space — quantile-scale each dimension,
as most VLA pipelines do. Reading the environment source first showed why
that would have been wrong: Franka Kitchen's action space **already is**
joint velocity in `[−1, 1]`, integrated into a position setpoint by the
environment itself.

Adding quantile scaling would have been a second normalization on top of one
that already happened. So there is none. The head emits nine numbers in the
action space the env accepts, and the env clips.

<div class="note">
<span class="note-label">Method</span>
This was checked by reading <code>kitchen_env.py</code>, not by assuming. The
whole decision took five minutes and removed a component from the design.
Checking the environment's actual contract before designing around it is the
cheapest step in the entire build.
</div>

## k = 1 first, chunking second

Action chunking — predicting the next k actions in one forward pass — is the
standard answer to compounding error over a long episode, and it is clearly
where this should go.

It is also a reshape, a change to how batches are sampled, a change to what
the head outputs, and an open question about what you do with the k−1 actions
you predicted but haven't executed yet. Every one of those can hide a bug in
the prefix, which is the part that is actually novel here.

So: k = 1, get it to zero loss on ten samples, confirm the arm moves, *then*
chunk. Diagnosability first.

<div class="note">
<span class="note-label">In hindsight</span>
Three of these four held up completely. The fourth — k = 1 — is still
correct as a starting point, but <a href="../10-rollouts/">Chapter 10</a>
shows the policy failing in a way that looks exactly like compounding error
on the long-horizon tasks, so chunking is now the most interesting lever
left rather than a nice-to-have.
</div>
