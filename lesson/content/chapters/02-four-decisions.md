---
title: "Four decisions before any code"
part: "Part I · Orientation"
chapter: 2
weight: 2
standfirst: "Each one closes a door. They are much cheaper to get right now than to reverse once something is trained."
---

Before we touch the model, there are four choices to make. None of them is
obviously correct, each one forecloses something later, and all four are far
cheaper to settle now than after a checkpoint depends on them.

## Should we cache the VLM's output?

The vision-language model is frozen, so its output for a given (frame,
instruction) pair never changes. We could run it once over all 56,005
demonstration steps, store the resulting vectors, and train the head on those
instead. Training would stop being a matter of hours and start being a matter of
seconds.

We're not going to, and the reason comes back three more times in this guide, so
it's the one thing to carry out of this chapter. Caching the output permanently forecloses
putting anything *into* the sequence. The moment we want a state token carrying
joint positions, or learned query tokens for predicting several actions at once,
or to unfreeze one layer of the backbone, every cached vector becomes garbage.

> Freezing the weights saves you the backward pass. Caching the outputs saves
> you the backward pass and costs you the architecture's future.

The cache does come back in Chapter 12, but as a throwaway diagnostic with a
known expiry date rather than as the path training runs on.

## Should the model emit tokens or numbers?

OpenVLA and several models after it discretize: they chop each action dimension
into 256 buckets, map those buckets onto vocabulary entries the model isn't
using, and let the existing language-model head predict them. It's tidy, and it
adds no new parameters at all.

Two costs rule it out for us. Emitting one action takes nine sequential decode
steps rather than one forward pass. And the loss becomes a cross-entropy over
buckets, which has a floor it can never go below.

That second cost sounds abstract, so here's what it buys us to avoid it.
Chapter 11 builds a test that asks whether the head can drive the loss on ten
samples to *exactly zero*. A number that can only ever reach 0.03 tells you far
less than one that reaches 0.000000, and we'd rather keep the unambiguous
version of our only end-to-end check.

## Do we need to normalize the actions?

Most VLA pipelines quantile-normalize the action space before training, and it's
close to reflex at this point.

Before adding one, we read the environment's source. Franka Kitchen interprets
your nine numbers as joint velocities in `[−1, 1]` and integrates them into a
position setpoint itself, which means the action space is already normalized and
already bounded. Adding quantile scaling would put a second normalization on top
of one that had already happened.

So there isn't one. The head emits nine raw numbers, and the environment clips
anything out of range. Five minutes of reading removed a component from the
design, which is a good trade whenever you can get it.

## One action at a time, or several?

*Action chunking* means predicting the next k actions in a single forward pass
rather than one. It's the standard answer to compounding error, where a small
mistake puts the arm somewhere slightly unfamiliar, which produces a slightly
worse action, which over three hundred steps becomes a policy waving at a
cabinet it can no longer reach.

It's also a reshape, a change to how batches are sampled, a change to the head's
output width, and an open question about what you do with the k−1 actions you
predicted but haven't executed yet. Any one of those can hide a bug in the
prefix, which is the part of this build that's new.

So we'll start with k = 1, drive it to zero loss on ten samples, confirm the arm
moves, and only then consider chunking.

<div class="note">
<span class="note-label">How these held up</span>
Three of the four were still right at the end. The fourth is right as a starting
point and wrong as a destination: by Chapter 13 the policy is failing on the
long-horizon tasks in a way that looks a great deal like compounding error, and
chunking has gone from a nice-to-have to the most interesting thing left to try.
</div>

Next: the environment, and what one observation contains.
