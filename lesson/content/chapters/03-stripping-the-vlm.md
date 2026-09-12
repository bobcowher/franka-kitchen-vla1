---
title: "Choosing and stripping the VLM"
part: "Part II · The model"
chapter: 3
weight: 3
standfirst: "A 500M model, and the one-line change that makes it 14× cheaper to run."
---

The model is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. 507,482,304
parameters, a 960-dimensional text stream, 32 decoder layers, a vision tower
that wants 512px inputs at patch 16.

It was picked for being small enough to run in the training inner loop on one
GPU while still being a real instruction-tuned VLM. That second half matters:
the whole premise is that language conditioning comes for free from
pretraining, and that requires a model that was actually trained to follow
instructions.

## Drop the vocabulary projection

`AutoModelForImageTextToText.from_pretrained(...)` gives you the full
generation model, including a `49280 × 960` output projection that maps
hidden states onto vocabulary logits.

We never generate text. That matrix is 47M parameters of pure waste in
memory and in the forward pass.

```python
self.vlm = AutoModelForImageTextToText.from_pretrained(
    VLM, dtype=torch.bfloat16, attn_implementation="sdpa").model
```

The `.model` at the end is the whole fix — it reaches past the generation
wrapper to the backbone that produces `last_hidden_state`, which is the only
thing the head ever reads.

## The one line that matters most

SmolVLM2 defaults to splitting each image into a 4×4 grid plus a global
thumbnail: seventeen sub-images per frame. That default exists because the
model is good at reading small text in scanned documents, and reading small
text needs resolution.

A robot looking at a kitchen counter needs none of it.

```python
processor.image_processor.do_image_splitting = False
```

<dl class="stats">
  <div><dt>Splitting on</dt><dd>1,139</dd></div>
  <div><dt>Splitting off</dt><dd>79</dd></div>
  <div><dt>Reduction</dt><dd>14.4×</dd></div>
</dl>

One frame goes from 1,139 tokens to 79. Attention is quadratic in sequence
length, so this is the single largest performance lever in the entire build,
and it is one line of configuration.

<div class="trap">
<span class="note-label">Worth checking yourself</span>
Defaults in multimodal processors are tuned for the benchmark the model was
published against, which is almost never your task. Print the token count of
one real input before you design anything around it. Ours was 14× larger than
necessary and nothing would have reported that as a problem — only as a slow
training run.
</div>

## Keeping it frozen, and keeping it that way

```python
self.vlm.requires_grad_(False)
```

That is the easy half. The subtle half is that `nn.Module.train()` recurses
into children, so the moment anything calls `model.train()` — which the
training loop does after every rollout — the frozen tower flips back into
training mode. There is no dropout in this particular stack, so it happens to
be harmless here, but it is exactly the kind of silent state change that
matters the instant you swap in a backbone that does have dropout.

```python
def train(self, mode=True):
    # train() recurses, so the frozen tower needs pinning back to eval.
    super().train(mode)
    self.vlm.eval()
    return self
```
