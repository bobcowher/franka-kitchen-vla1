---
title: "Building a VLA by Hand"
---

This is a guide to building a vision-language-action model out of parts rather
than fine-tuning someone else's. We take a small pretrained vision-language
model, freeze it, write an action head ourselves, and train it to drive a Franka
arm around a simulated kitchen.

Every number on these pages was measured on this data and this hardware. That
includes several that turned out to be wrong, which are kept because working out
why they were wrong taught us more than the model did.

<dl class="stats">
  <div><dt>Frozen</dt><dd>460M</dd></div>
  <div><dt>Trained</dt><dd>21,129</dd></div>
  <div><dt>Prefix</dt><dd>79–84</dd></div>
  <div><dt>Demos</dt><dd>56,005</dd></div>
  <div><dt>Best task</dt><dd>96%</dd></div>
</dl>

Start at Chapter 1 if you want to build along. If you would rather read one
chapter to decide whether the rest is worth your time, read
[Chapter 13]({{< relref "chapters/13-rollouts" >}}), which is about how we spent
a night watching a number that could not have told us anything.
