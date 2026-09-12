---
title: "The fast loop"
part: "Part IV · Knowing whether it works"
chapter: 9
weight: 9
standfirst: "The frozen model's output never changes. That fact turns a three-hour experiment into a sixty-second one."
---

Because the VLM is frozen, its output for a given (frame, instruction) pair is
a constant. Encode a sample of the dataset once, keep the vectors, and every
question about the *head* can be answered without ever running the VLM again.

```text
python scripts/probe.py --encode    # ~60s, writes checkpoints/probe_encodings.pt
python scripts/probe.py             # fits head variants in seconds
```

Anything that changes only the head, the readout, the loss weighting or the
normalization goes here first. The table in
[Chapter 5](../05-reading-the-prefix/) — four architectures, five seeds each,
twenty fits — is minutes of work. As a training run it would be a day.

<div class="note">
<span class="note-label">Scope, and its expiry</span>
This is the cache that <a href="../02-four-decisions/">Chapter 2</a> refused
to train on, reintroduced as a diagnostic with a known expiry date. It is
valid exactly as long as the VLM is frozen and the prompt is unchanged. The
moment a layer unfreezes, every vector in that file is stale and the harness
has to re-encode.
</div>

## Held out by shard, not by step

The 56,005 steps come from 582 recorded episodes. Consecutive steps within an
episode are nearly identical — the arm has moved a few millimeters. Splitting
randomly by step puts near-duplicates of training frames in the validation
set, and validation loss becomes an optimistic fiction.

So the split is by shard. A whole episode is either training or held out.

## One seed is not a measurement

This is the lesson the harness exists to enforce, and it was learned by
getting it wrong.

An early version fit each variant once. The fused head read **0.0934** in one
run and **0.1197** in another — on *identical cached data*, differing only in
initialization. That is a 28% swing, comfortably larger than the differences
between the architectures being compared. A confident claim had already been
written into a commit message on the strength of the first number.

<div class="finding">
<span class="note-label">Rule</span>
Repeat every fit across five seeds and report mean, best and worst. If the
spread between seeds is wider than the gap between your variants, you have not
measured anything — and you cannot know that from a single number.
</div>

The shipped configuration survived the correction: fused was genuinely best at
0.0905 mean, and it had the tightest spread of any variant. But it survived as
a measurement rather than as a lucky draw, which is a different epistemic
object.

This same failure — reading a small sample as if it were a measurement — is
about to happen again at a much larger scale, and in a place where it costs an
entire night. That is the next chapter.
