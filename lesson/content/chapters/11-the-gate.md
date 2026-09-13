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

Spread those ten across the episode rather than taking the first ten steps,
which are nearly the same picture and would make this question look worse than
it is:

```python
index = np.linspace(0, len(data["action"]) - 1, N).astype(int)
images = frames.resize(data["camera_scene"][index], IMAGE_SIZE)
```

<p class="filename">Filename: <strong>scripts/overfit.py</strong></p>

```python
def hidden(frames_u8, task_ids):
    """The vector the head sees, before the head."""
    with torch.no_grad():
        out = model.vlm(input_ids=model.prompt_ids[task_ids],
                        attention_mask=model.prompt_mask[task_ids],
                        pixel_values=model.preprocess(frames_u8))
    return out.last_hidden_state[torch.arange(len(task_ids), device="cuda"),
                                 model.prompt_end[task_ids]].float()


def spread(h, label):
    """How much of the hidden state actually varies across these inputs.

    Not cosine similarity. SmolLM2's residual stream has massive activation
    outliers -- one dim here carries sixty times the median magnitude -- so
    every pair reads ~0.99 whether or not the input mattered. Split
    h = mean + deviation and report the ratio instead.
    """
    deviation = (h - h.mean(0)).norm(dim=1).mean()
    print(f"  {label}: ||mean|| {h.mean(0).norm():.1f}  "
          f"||deviation|| {deviation:.2f}  ratio {deviation / h.mean(0).norm():.4f}")


print("
1. ten different frames, same instruction")
spread(hidden(images, task), "hidden state")
```

**Reading: 0.0515.** Small, but real.

## 2. Does the instruction move it?

One frame, all seven instructions. Same decomposition. This is language
conditioning existing at all, which is the entire justification for putting a
language model in the loop instead of a vision encoder.

```python
print("
2. one frame, all seven instructions")
all_tasks = torch.arange(len(TASK_DESCRIPTIONS)).cuda()
one_frame = np.repeat(images[:1], len(TASK_DESCRIPTIONS), axis=0)
spread(hidden(one_frame, all_tasks), "hidden state")
```

**Reading: 0.1225.** Larger than the image's effect.

## 3. Can ten samples reach zero loss?

```python
print(f"
3. overfitting {N} samples for {STEPS} steps")
optimizer = torch.optim.Adam(model.head.parameters(), 1e-2)
for step in range(STEPS):
    loss = F.mse_loss(model(images, task), actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == STEPS - 1:
        print(f"  step {step:>4}  loss {loss.item():.6f}")

print(f"
  target variance (loss if it predicted the mean): "
      f"{actions.var(0, unbiased=False).mean().item():.6f}")
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
