---
title: "Traps"
part: "Part V · What we learned"
chapter: 11
weight: 11
standfirst: "Six failures that each cost real time, and what makes them a category rather than a list."
---

Every one of these was found the expensive way. They are collected here
because they share a shape: **the system reported success while doing
something else**, and the only thing that caught them was an independent
measurement.

## Cosine similarity on a model with massive activations

Read 0.993–0.999 across ten genuinely different frames and looked like total
failure. A few dimensions with ~60× the median magnitude dominate every dot
product, so cosine reads ~0.99 between any two hidden states regardless of
input. Decompose into `mean + deviation` and compare norms instead.
See [Chapter 6](../06-the-action-head/).

## The wrong GPU, for four consecutive runs

CUDA defaults to `CUDA_DEVICE_ORDER=FASTEST_FIRST`. `nvidia-smi`, NVML and
every UI use PCI bus order. On a host where those disagree, an index that is
correct in one enumeration selects a different card in the other.

Four runs were dispatched to a 3090, logged "GPU 1", and every one executed on
a 3060 while the 3090 sat idle at 2 MiB.

| `CUDA_VISIBLE_DEVICES` | `CUDA_DEVICE_ORDER` | resolves to |
|---|---|---|
| 0 | default | RTX 3090 |
| 1 | default | RTX 3060 |
| 0 | `PCI_BUS_ID` | RTX 3060 |
| 1 | `PCI_BUS_ID` | RTX 3090 |

<div class="trap">
<span class="note-label">Trap · verify by UUID, never by index</span>
<p>Three attempted fixes failed before the cause was found, because each was
verified against a log line that printed the <em>intended</em> device rather
than the actual one. A banner that states intent as though it were outcome is
what made this expensive.</p>
<pre><code>nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid --format=csv,noheader</code></pre>
<p>The UUID is the only identifier that means the same thing in both
enumerations.</p>
</div>

## Axis order across the train/eval boundary

Removing an NCHW transpose from the dataset sampler but not from the
environment wrapper. It crashed on the first rollout, which was luck — with
compatible numbers it would have trained happily on transposed pixels.
See [Chapter 8](../08-the-gate/).

## One seed read as a measurement

The same head architecture scored 0.0934 and 0.1197 on identical cached data.
A claim had already been committed on the first number. Five seeds, always.
See [Chapter 9](../09-the-fast-loop/).

## Three rollouts read as a measurement

The same error, two orders of magnitude more expensive, over a whole night.
See [Chapter 10](../10-rollouts/).

## `pkill -f` matching its own shell

`pkill -f "scripts/evaluate.py"` matches the shell whose command line contains
that string — including the very command doing the killing, and any watcher
process polling for the same pattern. It reports success and exits non-zero
having killed the wrong things.

Record PIDs at launch and signal those, or verify with `ps` afterward. Never
trust the exit code of a pattern kill.

---

## The common shape

Five of these six are the same failure wearing different clothes:

> **A reading was accepted as evidence when the instrument could not have
> produced different output.**

Cosine similarity could not have distinguished a working model from a broken
one. The launch banner could not have reported the wrong GPU. Nine rollouts
could not have separated a 35% policy from a 60% one. One seed could not have
told you the variance was larger than the effect.

The defense is not care. It is asking, before trusting a number: *what reading
would I have gotten if the thing I am testing were false?* If the answer is
"the same one," the measurement is decoration.
