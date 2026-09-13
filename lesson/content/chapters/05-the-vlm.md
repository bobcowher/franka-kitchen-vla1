---
title: "Loading and stripping the VLM"
part: "Part II · The pieces"
chapter: 5
weight: 5
standfirst: "Three lines of configuration, one of which makes the model fourteen times cheaper to run."
---

Let's load the vision-language model our policy will be built on. We're using
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, which has a 960-dimensional text
stream, 32 decoder layers, and a vision tower that wants 512-pixel inputs.

We picked it for being small enough to run inside the training loop while still
being a real instruction-tuned VLM. That second half matters more than it might
seem: the whole premise of this build is that language conditioning arrives free
from pretraining, and that only works with a model someone trained to follow
instructions.

## Loading the model and dropping what we don't need

The obvious way to load it gives us more than we want:

<p class="filename">Filename: <strong>model.py</strong></p>

```python
VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

full = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.bfloat16)
print(f"{sum(p.numel() for p in full.parameters()):,}")
print(f"{sum(p.numel() for p in full.model.parameters()):,}")
```

<div class="output"><p class="output-label">This prints</p>

```text
507,482,304
460,173,504
```
</div>

The gap between those two numbers is a `49280 × 960` projection that maps hidden
states onto vocabulary logits, which is how the model would pick its next word
if we were asking it to write. We never generate text, so that matrix is 47
million parameters of memory and compute we'd carry for nothing.

Reaching past it is a single attribute. `full.model` is the backbone that
produces `last_hidden_state`, and that hidden state is the only thing our action
head will ever read:

<p class="listing">Listing 5.1 <em>Loading the frozen backbone</em></p>
<p class="filename">Filename: <strong>model.py</strong></p>

```python
VLM_IMAGE_SIZE = 512   # vision_config.image_size
HIDDEN = 960           # text_config.hidden_size

# .model drops the unused 49280 x 960 vocabulary projection.
self.vlm = AutoModelForImageTextToText.from_pretrained(
    VLM, dtype=torch.bfloat16, attn_implementation="sdpa").model
self.vlm.requires_grad_(False)
```

Two other arguments are doing work there. `dtype=torch.bfloat16` runs the
backbone in 16 bits, which is where the ~3 GB VRAM figure comes from; our head
will run in float32, and Chapter 9 shows where the conversion happens.
`attn_implementation="sdpa"` selects PyTorch's fused scaled-dot-product
attention rather than the slower eager version.

## The line that matters most

Now let's look at how the model turns an image into tokens, because the default
is wildly wrong for us. We'll build one prompt and see what comes back:

```python
proc = AutoProcessor.from_pretrained(VLM)
dummy = np.zeros((512, 512, 3), dtype=np.uint8)
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Open the microwave door"},
]}]
text = proc.apply_chat_template(messages, add_generation_prompt=True)

tokens = proc(text=[text], images=[[dummy]], return_tensors="pt")
print("input_ids   ", tuple(tokens["input_ids"].shape))
print("pixel_values", tuple(tokens["pixel_values"].shape))
```

<div class="output"><p class="output-label">This prints</p>

```text
input_ids    (1, 1139)
pixel_values (1, 17, 3, 512, 512)
```
</div>

Seventeen sub-images, and 1,139 tokens for a single frame. SmolVLM2 splits each
image into a 4×4 grid plus a global thumbnail, which is a sensible default for
the job it was benchmarked on: reading small text in scanned documents needs all
the resolution it can get. A robot looking at a kitchen counter needs none of
it.

We turn it off with one line, and run exactly the same code again:

<p class="filename">Filename: <strong>model.py</strong></p>

```python
processor = AutoProcessor.from_pretrained(VLM)
# On, a frame is 17 sub-images and 1139 tokens. Off, 79-84.
processor.image_processor.do_image_splitting = False
```

<div class="output"><p class="output-label">Now the same two prints give</p>

```text
input_ids    (1, 79)
pixel_values (1, 1, 3, 512, 512)
```
</div>

One sub-image, and 79 tokens. Attention cost grows with the square of sequence
length, so going from 1,139 to 79 is by a wide margin the largest performance
lever in this whole build, and it's a single line of configuration.

<div class="trap">
<span class="note-label">Worth checking on your own model</span>
Defaults in multimodal processors are tuned for whatever benchmark the model was
published against, which is almost never your task. Printing the token count of
one real input takes a minute. Ours was fourteen times larger than it needed to
be, and nothing anywhere would have reported that as a problem, only as a
training run that felt slow.
</div>

## Keeping the backbone frozen

`requires_grad_(False)` handles the gradients, and that part is easy. The part
that catches people is that `nn.Module.train()` recurses into its children, so
the moment anything calls `model.train()` the frozen tower flips back into
training mode along with everything else. Our training loop calls it after every
rollout.

The fix is to override `train` and pin the backbone back:

<p class="listing">Listing 5.2 <em>Keeping the VLM in eval mode through train() calls</em></p>
<p class="filename">Filename: <strong>model.py</strong></p>

```python
def train(self, mode=True):
    # train() recurses, so the frozen tower needs pinning back to eval.
    super().train(mode)
    self.vlm.eval()
    return self
```

This particular stack has no dropout, so leaving it out happens to be harmless
here. It stops being harmless the moment you swap in a backbone that does have
dropout, and at that point the symptom is a policy that behaves differently in
training and rollout for no visible reason.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
<p>Print the parameter count of your frozen backbone. You should see
<strong>460,173,504</strong>. If you see 507,482,304 you've kept the generation
wrapper and the <code>.model</code> suffix is missing.</p>
<p>Then feed one frame through your processor and print
<code>input_ids.shape</code>. You want 79, not 1,139.</p>
</div>

<div class="exercise">
<h4>Exercise 5.1 &nbsp;What splitting actually costs</h4>
<p>Time a single forward pass at batch 8 with <code>do_image_splitting</code> on
and then off, using <code>torch.cuda.synchronize()</code> either side of the
call so you're timing the GPU rather than the queue. The token counts differ by
14.4×. Does the time?</p>
<p>The answer is not the ratio you'd predict from the token count alone, and the
reason is worth working out before Chapter 9, where we profile the training step
and find the bottleneck somewhere else entirely.</p>
</div>

Next, we'll build the prompts. There are only seven instructions and they never
change, which lets us tokenize once at construction, but it also introduces a
padding problem that has no loud failure mode.
