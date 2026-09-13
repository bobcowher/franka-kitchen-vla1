---
title: "The prompt"
part: "Part II · The pieces"
chapter: 6
weight: 6
standfirst: "Seven instructions, tokenized once, and a padding bug with no loud failure mode."
---

There are seven tasks and therefore seven instructions. They never change.

<p class="filename">Filename: <strong>tasks.py</strong></p>

```python
TASKS = {
    "slide cabinet": "Slide the cabinet door open",
    "kettle":        "Move the kettle to the top left burner",
    "light switch":  "Turn on the overhead light switch",
    "microwave":     "Open the microwave door",
    "bottom burner": "Turn the oven knob for the bottom left burner",
    "top burner":    "Turn the oven knob for the top left burner",
    "hinge cabinet": "Open the cabinet second from the left",
}

# Dict order is what task_index()'s integer ids mean, so reordering TASKS
# silently invalidates every checkpoint trained before the change.
TASK_DESCRIPTIONS = list(TASKS.values())
_TASK_INDEX = {description: i for i, description in enumerate(TASK_DESCRIPTIONS)}
```

That comment earns its place. The integer task id is a position in a Python
dict. Reorder the dict and the model maps "open the microwave" onto weights
learned for a different task, with nothing raising.

## Tokenize once

The text side of the input is a pure function of the task id, so build all
seven prompts at construction and index them per batch.

<p class="filename">Filename: <strong>model.py</strong></p>

```python
def _build_prompts(self, processor):
    """Tokenize the seven instructions once; they never change."""
    prompts = []
    for description in TASK_DESCRIPTIONS:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": description},
        ]}]
        prompts.append(processor.apply_chat_template(
            messages, add_generation_prompt=True))

    dummy = np.zeros((VLM_IMAGE_SIZE, VLM_IMAGE_SIZE, 3), dtype=np.uint8)
    tokens = processor(text=prompts, images=[[dummy]] * len(prompts),
                       padding=True, return_tensors="pt")
```

The dummy image is there because the processor needs an image to expand the
`{"type": "image"}` placeholder into real image-token positions. You throw the
pixels away; only the token layout survives.

This is not the prefix caching [Chapter 2]({{< relref "chapters/02-four-decisions" >}}) rejected. No
VLM output is stored, only token ids, so unfreezing the model later invalidates
none of it.

## The padding problem

"Open the microwave door" is 79 tokens. "Turn the oven knob for the bottom left
burner" is 84. A batch mixing tasks has to pad, and padding collides
immediately with a head that reads the last position.

<div class="trap">
<span class="note-label">Trap</span>
<p>Right-pad and read <code>last_hidden_state[:, -1]</code> and you read a
<strong>pad token</strong> for every row shorter than the longest in the batch.
Nothing raises. Nothing warns. The model learns worse for five of seven tasks
and the loss curve looks entirely normal.</p>
<p>Left-padding looks like the fix and is a different bug. <em class="term">RoPE</em>
— rotary position embedding, how this model encodes where each token sits in
the sequence — is computed from absolute index. Left-padding shifts every
index, so the model sees the image at positions it was never pretrained on.</p>
</div>

Right-pad, and gather each row's last real token:

<p class="filename">Filename: <strong>model.py</strong></p>

```python
self.register_buffer("prompt_ids", tokens["input_ids"], persistent=False)
self.register_buffer("prompt_mask", tokens["attention_mask"], persistent=False)
# Right-padded, so the last real token is at mask.sum() - 1. Left
# padding would shift every RoPE position.
self.register_buffer("prompt_end", tokens["attention_mask"].sum(1) - 1,
                     persistent=False)
```

Attention here is <em class="term">causal</em>: each position attends only to
positions before it, never after. So a real token cannot see a later pad token,
and the gathered hidden state is bit-identical to what you would get running
that sequence unpadded. No masking subtleties, no position shift.

`persistent=False` keeps these out of the saved state dict. They are derived
from the tokenizer, not learned.

## Locating the image tokens

The next chapter needs to know which positions hold image content. The
processor expands one `<image>` placeholder into 64 real image tokens, so find
them by id.

<p class="filename">Filename: <strong>model.py</strong></p>

```python
self.register_buffer(
    "image_positions",
    (tokens["input_ids"] == processor.tokenizer.convert_tokens_to_ids(
        "<image>")).unsqueeze(-1), persistent=False)
```

The `unsqueeze(-1)` adds a trailing dimension so the mask broadcasts over the
960 features when you use it.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
Print <code>prompt_mask.sum(1)</code>. You should see seven numbers between 79
and 84, not seven identical ones. Then print
<code>image_positions.sum(1)</code> and confirm 64 for every task.
</div>
