---
title: "Loading and stripping the VLM"
part: "Part II · The pieces"
chapter: 5
weight: 5
standfirst: "Three lines of configuration, one of which makes the model fourteen times cheaper to run."
---

The model is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. 507,482,304
parameters, a 960-dimensional text stream, 32 decoder layers, and a vision
tower that wants 512-pixel inputs.

Pick it for being small enough to run inside the training loop while still
being a real instruction-tuned VLM. That second half is not optional: the whole
premise is that language conditioning arrives free from pretraining, and that
requires a model trained to follow instructions.

## Load it, and drop the vocabulary projection

<p class="filename">Filename: <strong>model.py</strong></p>

```python
VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_IMAGE_SIZE = 512   # vision_config.image_size
HIDDEN = 960           # text_config.hidden_size

# .model drops the unused 49280 x 960 vocabulary projection.
self.vlm = AutoModelForImageTextToText.from_pretrained(
    VLM, dtype=torch.bfloat16, attn_implementation="sdpa").model
self.vlm.requires_grad_(False)
```

Three things happen in that call.

`AutoModelForImageTextToText` normally returns the full generation model,
including a 49280 × 960 output projection that maps hidden states onto
vocabulary <em class="term">logits</em> — the raw scores for each possible next
token. You never generate text. The `.model` suffix reaches past the generation
wrapper to the backbone that produces `last_hidden_state`, which is the only
thing the head reads. That saves 47M parameters of memory and compute.

`dtype=torch.bfloat16` runs the model in 16-bit. The head runs in float32;
Chapter 9 shows where the conversion happens.

`attn_implementation="sdpa"` uses PyTorch's fused scaled-dot-product attention
rather than the eager implementation.

## The line that matters most

SmolVLM2 splits each image into a 4×4 grid plus a global thumbnail, giving
seventeen sub-images per frame. That default exists because the model is good
at reading small text in scanned documents, and small text needs resolution.

A robot looking at a kitchen counter needs none of it.

<p class="filename">Filename: <strong>model.py</strong></p>

```python
processor = AutoProcessor.from_pretrained(VLM)
# On, a frame is 17 sub-images and 1139 tokens. Off, 79-84.
processor.image_processor.do_image_splitting = False
```

<dl class="stats">
  <div><dt>Splitting on</dt><dd>1,139</dd></div>
  <div><dt>Splitting off</dt><dd>79</dd></div>
  <div><dt>Reduction</dt><dd>14.4×</dd></div>
</dl>

Attention cost grows with the square of sequence length, so this is the largest
single performance lever in the build, and it is one line of configuration.

<div class="trap">
<span class="note-label">Check this yourself</span>
Defaults in multimodal processors are tuned for the benchmark the model was
published against, which is almost never your task. Print the token count of
one real input before designing anything around it. Ours was 14× larger than
necessary, and nothing would have reported that as a problem — only as a slow
training run.
</div>

## Keeping it frozen

`requires_grad_(False)` is the easy half. The subtle half is that
`nn.Module.train()` recurses into children. The moment anything calls
`model.train()`, which the training loop does after every rollout, the frozen
tower flips back into training mode.

<p class="filename">Filename: <strong>model.py</strong></p>

```python
def train(self, mode=True):
    # train() recurses, so the frozen tower needs pinning back to eval.
    super().train(mode)
    self.vlm.eval()
    return self
```

This stack has no dropout, so the bug is harmless here. It stops being harmless
the instant you swap in a backbone that does.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
Print <code>sum(p.numel() for p in self.vlm.parameters())</code> and confirm you
see 507,482,304 rather than the ~555M you get with the vocabulary projection
attached. Then feed one frame through the processor and print the resulting
sequence length. It should be 79, not 1,139.
</div>
