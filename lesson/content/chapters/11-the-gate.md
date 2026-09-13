---
title: "The gate"
part: "Part III · Knowing whether it works"
chapter: 11
weight: 11
standfirst: "Three questions in an order where each only matters if the last one passed, answered in under a minute."
---

`scripts/overfit.py` is our entire test suite. It asks three questions, and the
order is the whole point: a failure in an earlier one makes the later answers
meaningless, so we ask them cheapest first and stop at the first no.

Let's run the finished thing before walking through it:

<div class="output"><p class="output-label"><code>python scripts/overfit.py</code></p>

```text
dataset/microwave/shard_2026_08_08_10_25_18.npz: 'Open the microwave door',
steps [0, 8, 16, 24, 32, 41, 49, 57, 65, 74]

1. ten different frames, same instruction
  hidden state: ||mean|| 53.8  ||deviation|| 2.77  ratio 0.0515

2. one frame, all seven instructions
  hidden state: ||mean|| 53.1  ||deviation|| 6.50  ratio 0.1225

3. overfitting 10 samples for 400 steps
  step    0  loss 0.693121
  step   50  loss 0.149218
  step  100  loss 0.038258
  step  150  loss 0.014115
  step  200  loss 0.003616
  step  250  loss 0.000589
  step  300  loss 0.000062
  step  350  loss 0.000005
  step  399  loss 0.000000

  target variance (loss if it predicted the mean): 0.186505
```
</div>

About fifty seconds, most of which is loading the model. Now let's see how each
of those three numbers is produced.

## Picking the ten frames

<p class="filename">Filename: <strong>scripts/overfit.py</strong></p>

```python
# Spread across the episode rather than the first ten steps, which are nearly
# the same picture and would make question 1 look worse than it is.
index = np.linspace(0, len(data["action"]) - 1, N).astype(int)
images = frames.resize(data["camera_scene"][index], IMAGE_SIZE)
```

That's why the output says `steps [0, 8, 16, ... 74]` rather than 0 through 9.
Ten consecutive frames of a robot arm differ by a few millimetres of travel, and
using them would tell us the image barely changes the hidden state when really
we'd just picked ten nearly identical images.

## Question one: does the image move the hidden state?

If ten different frames produce the same vector, our head has nothing to read.

<p class="listing">Listing 11.1 <em>Reading the head's input, and measuring how much of it varies</em></p>
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


print("\n1. ten different frames, same instruction")
spread(hidden(images, task), "hidden state")
```

Chapter 8 covered why this isn't cosine similarity. **0.0515** is small but real,
and it's also the number that tells us the head is going to need a `LayerNorm`.

## Question two: does the instruction move it?

<p class="filename">Filename: <strong>scripts/overfit.py</strong></p>

```python
print("\n2. one frame, all seven instructions")
all_tasks = torch.arange(len(TASK_DESCRIPTIONS)).cuda()
one_frame = np.repeat(images[:1], len(TASK_DESCRIPTIONS), axis=0)
spread(hidden(one_frame, all_tasks), "hidden state")
```

One picture, seven different sentences. **0.1225**, which is 2.4 times the
image's effect. This is language conditioning existing at all, and it's the
entire justification for putting a language model in the loop instead of a
vision encoder.

## Question three: can ten samples reach zero?

<p class="listing">Listing 11.2 <em>Overfitting ten samples</em></p>
<p class="filename">Filename: <strong>scripts/overfit.py</strong></p>

```python
print(f"\n3. overfitting {N} samples for {STEPS} steps")
optimizer = torch.optim.Adam(model.head.parameters(), 1e-2)
for step in range(STEPS):
    loss = F.mse_loss(model(images, task), actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == STEPS - 1:
        print(f"  step {step:>4}  loss {loss.item():.6f}")

print(f"\n  target variance (loss if it predicted the mean): "
      f"{actions.var(0, unbiased=False).mean().item():.6f}")
```

Watch the shape of that descent in the output above: 0.69, then 0.15, then a
long smooth slide to 0.000000. That tells us the shapes line up, gradient
reaches the weights, and Adam can drive the objective.

The last line is there to keep us honest. If the head simply predicted the mean
action it would score 0.186505, so 0.000000 is a real fit rather than a
degenerate one.

<div class="note">
<span class="note-label">What this proves, and what it doesn't</span>
It proves the machine is wired together. It proves <em>nothing whatsoever</em>
about generalization, since a model that memorizes ten samples perfectly may be
useless on the eleventh. Treat it as a compile check that happens to return a
loss value.
</div>

## Why a gate rather than unit tests

Unit tests here would mostly assert tensor shapes, and the framework already
checks shapes at runtime. The failures that actually happen in a build like this
one are semantic: reading a pad token, transposed pixels, a frozen module left in
training mode, a task id pointing at the wrong instruction. Not one of those
changes a shape.

What catches them is an end-to-end signal with a known correct value, and zero is
such a value. That's the whole reason Chapter 8 gave up the `tanh`.

<div class="trap">
<span class="note-label">Trap · run the rollout path too</span>
<p>The gate exercises the training path and nothing else. When we removed the
NCHW transpose from the dataset without removing it from the environment
wrapper, the gate passed happily and the first rollout died with:</p>
<pre><code>RuntimeError: expected input[1, 448, 512, 512] to have
3 channels, but got 448 channels instead</code></pre>
<p>That was the lucky outcome. The channel counts happened not to line up, so it
crashed at once. Had they been compatible, training would have run to completion
on transposed pixels and produced a worse policy with nothing reporting a
problem. One rollout costs seven seconds.</p>
</div>

<div class="exercise">
<h4>Exercise 11.1 &nbsp;Break it on purpose</h4>
<p>Change <code>prompt_end</code> to always read position <code>-1</code> and run
the gate again. Question 3 will still reach zero, because ten samples of one task
are all padded to the same length and there's no disagreement to expose.</p>
<p>Now make the ten samples span two tasks with different instruction lengths and
run it once more. Watching which questions still pass is a good lesson in what a
passing test does and does not cover.</p>
</div>

Next: a way to test head variants in seconds rather than hours.
