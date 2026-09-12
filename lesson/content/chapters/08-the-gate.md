---
title: "The gate"
part: "Part IV · Knowing whether it works"
chapter: 8
weight: 8
standfirst: "Three questions in dependency order, answered in under a minute, before any run longer than a coffee break."
---

`scripts/overfit.py` is the entire test suite. It asks three questions, and
the order is the point — a failure in an earlier one makes the later answers
meaningless.

## 1. Does the image move the hidden state?

Encode ten different frames with the same instruction. Decompose
`h = mean + deviation` and compare norms. If the deviation is zero, the head
has nothing to read and no amount of training helps.

**Reading: 0.0515.** Small, but real. (And *not* cosine similarity, for the
reason in [Chapter 6](../06-the-action-head/).)

## 2. Does the instruction move it?

Encode one frame with all seven instructions. Same decomposition. This is
language conditioning existing at all — the entire justification for putting
a language model in the loop rather than a vision encoder.

**Reading: 0.1225.** Bigger than the image's effect.

## 3. Can ten samples reach zero loss?

Take ten training samples and fit them until the loss stops moving. This
proves the shapes line up, the gradient reaches the weights, and the
optimizer can actually drive the objective.

**Reading: 0.000000** after 400 steps.

<div class="note">
<span class="note-label">What this does and does not prove</span>
It proves the machine is wired together. It proves <em>nothing whatsoever</em>
about generalization — a model that memorizes ten samples perfectly may be
useless on the eleventh. Treat it as a compile check with a loss value, not as
evidence the approach works.
</div>

## Why a gate instead of unit tests

Conventional unit tests on this code would mostly assert tensor shapes, and
shapes are already checked by the framework at runtime. The failures that
actually happen in this kind of build are semantic: reading a pad token,
transposed pixels, a frozen module quietly in training mode, a task id
pointing at the wrong instruction. None of those change a shape.

What catches them is an end-to-end signal that has a known correct value.
Zero is such a value. That is why [Chapter 6](../06-the-action-head/) gave up
the `tanh` to keep it reachable.

<div class="trap">
<span class="note-label">Trap · run the rollout path too</span>
<p>The gate exercises the training path. It does not exercise the environment
path, and those are different code — different observation shapes, different
call sites.</p>
<p>Removing an NCHW transpose from the dataset sampler without removing it
from the environment wrapper produced this, on the first rollout:</p>
<pre><code>RuntimeError: expected input[1, 448, 512, 512] to have
3 channels, but got 448 channels instead</code></pre>
<p>That is the <em>lucky</em> outcome — the channel count happened not to line
up, so it crashed immediately. Had the numbers been compatible, training would
have run to completion on transposed pixels and produced a worse policy with
nothing reporting a problem. One rollout costs seven seconds and is the only
thing that exercises the other half of the code.</p>
</div>
