---
title: "Four decisions before any code"
part: "Part I · Orientation"
chapter: 2
weight: 2
standfirst: "Each one closes a door. Getting them in the right order is most of the work."
---

Four choices come before the first line. None is obviously correct, all of
them foreclose something, and each is expensive to reverse once a trained
checkpoint depends on it.

## Live forward pass, not a cached prefix

The VLM is frozen, so its output for a given (frame, instruction) pair never
changes. You could run it once over all 56,005 demonstration steps, store the
resulting vectors, and train the head on those. Training would be enormously
faster.

Reject it, for one reason: caching the output permanently forecloses putting
anything *into* the sequence. The moment you want a state token for
proprioception, or learned query tokens for action chunking, or to unfreeze a
layer, every cached vector is garbage.

> Freezing the weights saves you the backward pass. Caching the outputs saves
> you the backward pass and costs you the architecture's future.

The cache returns in [Chapter 12]({{< relref "chapters/12-the-fast-loop" >}}), as a diagnostic with
a known expiry date rather than as the training path.

## Continuous output, not action tokens

OpenVLA and its descendants discretize. Each action dimension is binned into
256 buckets, the buckets are mapped onto unused vocabulary entries, and the
existing language-model head predicts them. It adds zero parameters.

Two costs rule it out here. It needs nine sequential decode steps to emit one
action. And its loss is a cross-entropy over buckets, which never reaches an
interpretable zero.

That second cost is the larger one. [Chapter 11]({{< relref "chapters/11-the-gate" >}}) is built
entirely on "can this reach exactly zero on ten samples," and a quantization
floor would blunt the only unambiguous diagnostic in the design.

## Joint velocity, emitted directly

Covered in [Chapter 3]({{< relref "chapters/03-the-environment" >}}): the environment's action space
already is joint velocity in `[−1, 1]`. So the head emits nine raw numbers and
the environment clips. No normalization layer, no unscaling at inference.

## k = 1 first, chunking second

<em class="term">Action chunking</em> means predicting the next k actions in
one forward pass instead of one. It is the standard answer to compounding
error, where a small mistake moves the arm somewhere slightly unfamiliar, which
produces a worse action, which compounds over a few hundred steps.

It is also a reshape, a change to batch sampling, a change to the head's output
width, and an open question about what you do with the k−1 actions you
predicted but have not executed. Any of those can hide a bug in the prefix,
which is the genuinely new part of this build.

So: k = 1, drive it to zero loss on ten samples, confirm the arm moves, then
chunk.

<div class="note">
<span class="note-label">In hindsight</span>
Three of these four held up. The fourth is still right as a starting point,
but <a href="{{< relref "chapters/13-rollouts" >}}">Chapter 13</a> shows the policy failing in a way
that looks exactly like compounding error on the long-horizon tasks. Chunking
is now the most interesting lever left rather than a nice-to-have.
</div>
