---
title: "Traps"
part: "Part IV · Lessons"
chapter: 14
weight: 14
standfirst: "Six failures that each cost real time, and the one shape they share."
---

Every one of these was found the expensive way. They share a shape: **the
system reported success while doing something else**, and only an independent
measurement caught it.

## Cosine similarity on a model with massive activations

Read 0.993 to 0.999 across ten genuinely different frames and looked like total
failure. A few dimensions carry about sixty times the median magnitude and
dominate every dot product, so cosine reads ~0.99 between any two hidden states
regardless of input. Decompose into `mean + deviation` and compare norms
instead. See [Chapter 8]({{< relref "chapters/08-the-action-head" >}}).

## The wrong GPU, for four consecutive runs

CUDA defaults to `CUDA_DEVICE_ORDER=FASTEST_FIRST`. `nvidia-smi`, NVML and
every UI use PCI bus order. Where those disagree, an index that is correct in
one enumeration selects a different card in the other.

Four runs were dispatched to a 3090, logged "GPU 1", and every one executed on
a 3060 while the 3090 sat idle at 2 MiB.

| `CUDA_VISIBLE_DEVICES` | `CUDA_DEVICE_ORDER` | resolves to |
|---|---|---|
| 0 | default | RTX 3090 |
| 1 | default | RTX 3060 |
| 0 | `PCI_BUS_ID` | RTX 3060 |
| 1 | `PCI_BUS_ID` | RTX 3090 |

The fix is to pin the order before CUDA initializes:

<p class="filename">Filename: <strong>scripts/train.py</strong></p>

```python
import os

# The 3090 is GPU 1 to nvidia-smi but GPU 0 to CUDA, which orders by speed
# unless told otherwise. Pinning the order makes the two agree.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

<div class="trap">
<span class="note-label">Trap · verify by UUID, never by index</span>
<p>Three attempted fixes failed before the cause was found, because each was
verified against a log line printing the <em>intended</em> device rather than
the actual one. A banner that states intent as though it were outcome is what
made this expensive.</p>
<pre><code>nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid --format=csv,noheader</code></pre>
<p>The UUID is the only identifier that means the same thing in both
enumerations.</p>
</div>

## Axis order across the train/eval boundary

An NCHW transpose removed from the dataset but not from the environment
wrapper. It crashed on the first rollout, which was luck. With compatible
numbers it would have trained happily on transposed pixels. See
[Chapter 11]({{< relref "chapters/11-the-gate" >}}).

## One seed read as a measurement

The same head scored 0.0934 and 0.1197 on identical cached data. A claim had
already been committed on the first number. Five seeds, always. See
[Chapter 12]({{< relref "chapters/12-the-fast-loop" >}}).

## Three rollouts read as a measurement

The same error, two orders of magnitude more expensive, across a whole night.
See [Chapter 13]({{< relref "chapters/13-rollouts" >}}).

## `pkill -f` matching its own shell

`pkill -f "scripts/evaluate.py"` matches every process whose command line
contains that string, including the shell running the `pkill` itself and any
watcher polling for the same pattern. It reports success, exits non-zero, and
kills the wrong things.

Record PIDs at launch and signal those:

```bash
for e in 27500 40000 42500; do
  python scripts/evaluate.py "checkpoints/e$e" --rollouts 50 &
  echo $! >> pids
done
while read p; do kill -0 "$p" 2>/dev/null && echo "alive: $p"; done < pids
```

Never trust the exit code of a pattern kill. Check `ps`.

## The common shape

Five of these six are one failure in different clothes:

> **A reading was accepted as evidence when the instrument could not have
> produced a different answer.**

Cosine similarity could not have distinguished a working model from a broken
one. The launch banner could not have reported the wrong GPU. Nine rollouts
could not have separated a 35% policy from a 60% one. One seed could not have
told you the variance exceeded the effect.

The defense is not care. Before trusting a number, ask: *what reading would I
have gotten if the thing I am testing were false?* If the answer is "the same
one," the measurement is decoration.
