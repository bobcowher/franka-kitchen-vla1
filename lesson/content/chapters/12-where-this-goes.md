---
title: "What we have, and what's next"
part: "Part V · What we learned"
chapter: 12
weight: 12
standfirst: "A working policy, a measured ceiling, and four levers in the order they become worth pulling."
---

## Where it landed

A frozen 507M-parameter vision-language model with a 21,129-parameter head
trained on 56,005 demonstration steps, driving a Franka arm from one camera
frame and one English sentence.

| task | success | 95% CI |
|---|---|---|
| slide cabinet | 96% | [86.5, 98.9] |
| hinge cabinet | 68% | [54.2, 79.2] |
| top burner | 54% | [40.4, 67.0] |
| bottom burner | 8% | [3.2, 18.8] |
| kettle | 0% | [0.0, 7.1] |
| light switch | 0% | [0.0, 7.1] |
| microwave | ~2% | (in-run, n≈120) |

Milestone met, comfortably. The interesting part is no longer whether it
works but the *shape* of the failure: three tasks performed, one marginal,
three untouched. A uniformly mediocre policy would suggest a capacity
problem. This does not — it suggests something task-specific.

## What the ceiling looks like

The evidence that the frozen representation, not the head, is the binding
constraint:

- A head with **47× more parameters** scores worse and is wildly unstable —
  its worst seed is exactly the predict-the-mean baseline.
- Held-out MSE bottoms out around **0.0905** against a 0.2132 baseline, so the
  readout explains roughly 58% of action variance and no head architecture
  tried moves it.
- Checkpoints **trade tasks against each other** rather than improving
  together, which is what interference looks like when a small head is asked
  to hold several behaviors over a fixed representation.

## The levers, in order

**1 · Unfreeze the top of the text stack.** The first real test of the ceiling
hypothesis. Implemented as `UNFREEZE_LAST_N_LAYERS`, which matches parameters
by name rather than hardcoding a module path, with unfrozen weights getting
their own optimizer group at a tenth of the head's learning rate — a
head-sized LR on pretrained weights destroys them quickly. Two layers is
19,664,640 parameters, a thousand times the head.

<div class="note">
<span class="note-label">Status</span>
Run 10 trained this for 20,000 epochs against a frozen control. Both arms are
indistinguishable on their in-run evaluation — which, after
<a href="../10-rollouts/">Chapter 10</a>, tells you nothing at all. The A/B is
being re-run offline at n=50 per checkpoint, which is the only version of the
comparison worth reading.
</div>

**2 · Diagnose the dead tasks.** Three tasks at flat zero is not a gradual
capability limit, it is something categorical. Candidates: too few
demonstrations for those tasks, an instruction that doesn't distinguish itself
in embedding space, or start-pose geometry that puts the goal outside the
camera frame. This is cheap to investigate and might be worth more than any
architecture change.

**3 · Action chunking, k = 8.** The standard answer to compounding error over
a long episode, and the failure mode on the long-horizon tasks looks like
exactly that. This is where the live forward pass from
[Chapter 2](../02-four-decisions/) finally earns itself: build embeddings from
`input_ids`, concatenate k learned query vectors, extend the attention mask by
k. Impossible with a cached prefix.

**4 · A proprioceptive state token.** `Linear(9 → 960)` in the prefix, the π₀
convention. Deliberately left out of the first build: with joint position
wired to the head, the model can partially fit while the visual prefix is
garbage, and [the gate](../08-the-gate/) loses its ability to tell you. It
rides in on the machinery chunking already needs.

## The test nobody has run yet

Everything above is about control. The reason to put a *language* model in the
loop is generalization over instructions, and that has never been tested —
because an `nn.Embedding(7)` would pass every conditioning check in this
guide.

The real test is a sentence the model never trained on. "Open the cabinet on
the left" instead of "Open the cabinet second from the left." If the arm still
moves correctly, the language model is doing something a lookup table cannot.
If it doesn't, this is an expensive seven-way classifier and the honest move
is to say so.

That is the first experiment the previous architecture could not have run, and
it is the one this whole build was for.

---

<div class="finding">
<span class="note-label">If you take one thing</span>
The architecture in this guide worked on day one. What cost time was never the
model — it was accepting readings from instruments that could not have
produced a different answer. Build the measurement with the same care as the
thing being measured, and compute its variance before you trust it.
</div>
