---
title: "Building a VLA by Hand"
---

This is a build log for a vision-language-action model assembled from parts,
not fine-tuned from someone else's. No SmolVLA, no lerobot, no adapting an
existing VLA. A frozen 507M-parameter vision-language model, an action head
written from scratch, and a Franka arm in a simulated kitchen.

The point is the construction. Every number on these pages was measured on
this data, on this hardware — including the ones that turned out to be wrong,
which are the more useful half.

<dl class="stats">
  <div><dt>Frozen</dt><dd>507M</dd></div>
  <div><dt>Trained</dt><dd>21,129</dd></div>
  <div><dt>Prefix</dt><dd>79–84</dd></div>
  <div><dt>Demos</dt><dd>56,005</dd></div>
  <div><dt>Best task</dt><dd>96%</dd></div>
</dl>

If you read nothing else, read [Chapter 10](chapters/10-rollouts/). The
architecture worked earlier than we could tell, and the reason we couldn't
tell had nothing to do with the architecture.
