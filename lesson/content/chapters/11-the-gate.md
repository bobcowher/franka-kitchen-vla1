---
title: "The gate"
part: "Part III · Knowing whether it works"
chapter: 11
weight: 11
standfirst: "Three questions in dependency order, answered in under a minute, before any run longer than a coffee break."
---

`scripts/overfit.py` is the whole test suite. Three questions, and the order is
the point: a failure in an earlier one makes the later answers meaningless.

## 1. Does the image move the hidden state?

Encode ten different frames with the same instruction. If they produce the same
vector, the head has nothing to read.

<p class="filename">Filename: <strong>scripts/overfit.py</strong></p>

```python
def signal_ratio(h):
    """How much of the hidden state varies across samples.

    Not cosine similarity: a few dimensions carry ~60x the median magnitude
    and dominate any dot product, so cosine reads ~0.99 regardless.
    """
    mean = h.mean(0, keepdim=True)
    return (h - mean).norm(dim=1).mean().item() / mean.norm().item()


frames = torch.stack([sample_frame() for _ in range(10)])
h = encode(frames, task=torch.zeros(10, dtype=torch.long))
print(f"image  ratio {signal_ratio(h):.4f}")
```

**Reading: 0.0515.** Small, but real.

## 2. Does the instruction move it?

One frame, all seven instructions. Same decomposition. This is language
conditioning existing at all, which is the entire justification for putting a
language model in the loop instead of a vision encoder.

```python
h = encode(frames[:1].expand(7, -1, -1, -1), task=torch.arange(7))
print(f"instr  ratio {signal_ratio(h):.4f}")
```

**Reading: 0.1225.** Larger than the image's effect.

## 3. Can ten samples reach zero loss?

```python
states, actions, _, _, tasks = dataset.sample_batch(10)
for step in range(400):
    loss = F.mse_loss(model(states['camera_scene'], tasks), actions)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
print(f"overfit 10 samples: {loss.item():.6f}")
```

**Reading: 0.000000** after 400 steps at lr 1e-2.

This proves the shapes line up, gradient reaches the weights, and the optimizer
can drive the objective. It proves nothing about generalization — a model that
memorizes ten samples may be useless on the eleventh. Treat it as a compile
check with a loss value.

## Why a gate instead of unit tests

Unit tests here would mostly assert tensor shapes, and the framework already
checks shapes at runtime. The failures that actually happen in this kind of
build are semantic: reading a pad token, transposed pixels, a frozen module in
training mode, a task id pointing at the wrong instruction. None of those
changes a shape.

What catches them is an end-to-end signal with a known correct value. Zero is
such a value, which is why [Chapter 8]({{< relref "chapters/08-the-action-head" >}}) gave up the
`tanh` to keep it reachable.

<div class="trap">
<span class="note-label">Trap · run the rollout path too</span>
<p>The gate exercises the training path only. Removing an NCHW transpose from
the dataset without removing it from the environment wrapper produced this on
the first rollout:</p>
<pre><code>RuntimeError: expected input[1, 448, 512, 512] to have
3 channels, but got 448 channels instead</code></pre>
<p>That is the lucky outcome. The channel count happened not to line up, so it
crashed immediately. Had the numbers been compatible, training would have run
to completion on transposed pixels and produced a worse policy with nothing
reporting a problem. One rollout costs seven seconds.</p>
</div>
