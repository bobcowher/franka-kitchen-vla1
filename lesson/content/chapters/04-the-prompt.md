---
title: "The prompt, and which token you read"
part: "Part II · The model"
chapter: 4
weight: 4
standfirst: "Seven instructions, tokenized once. Then the padding problem that has no loud failure mode."
---

There are seven tasks, and therefore seven instructions. They never change.

```text
slide cabinet   Slide the cabinet door open
kettle          Move the kettle to the top left burner
light switch    Turn on the overhead light switch
microwave       Open the microwave door
bottom burner   Turn the oven knob for the bottom left burner
top burner      Turn the oven knob for the top left burner
hinge cabinet   Open the cabinet second from the left
```

Since the text side of the input is a pure function of the task id, it is
tokenized once at construction and indexed per batch. This is *not* the
prefix caching that [Chapter 2](../02-four-decisions/) ruled out — no VLM
output is stored, only token ids — so unfreezing the model later invalidates
none of it.

```python
self.register_buffer("prompt_ids", tokens["input_ids"], persistent=False)
self.register_buffer("prompt_mask", tokens["attention_mask"], persistent=False)
```

<div class="note">
<span class="note-label">Ordering hazard</span>
The integer task id is the position in a Python dict. Reordering that dict
silently invalidates every checkpoint ever trained — the model would map
"open the microwave" onto the weights learned for a different task, with
nothing raising. This is worth a comment in the source, and it has one.
</div>

## The padding problem

"Open the microwave door" tokenizes to 79 tokens. "Turn the oven knob for the
bottom left burner" tokenizes to 84. A batch that mixes tasks has to pad, and
padding collides immediately with a head that wants to read the last position.

<div class="trap">
<span class="note-label">Trap</span>
<p>Right-pad and read <code>last_hidden_state[:, -1]</code> and you are reading
a <strong>pad token</strong> for every row shorter than the longest one in the
batch. Nothing raises. Nothing warns. The model simply learns worse for five
of the seven tasks, and the loss curve looks completely normal.</p>
<p>Left-padding looks like the obvious fix, and it is a different bug: it
shifts every RoPE position, so the model sees the image at absolute positions
it was never pretrained on.</p>
</div>

The clean answer is to right-pad and then *gather* each row's last real
token:

```python
self.register_buffer("prompt_end", tokens["attention_mask"].sum(1) - 1,
                     persistent=False)
...
last = h[torch.arange(len(task)), self.prompt_end[task]]
```

Under causal attention a real token cannot attend to a later pad token, so the
gathered hidden state is bit-identical to what you would get by running that
sequence unpadded. No masking subtleties, no position shift, no silent
degradation.

## Finding the image tokens

The other readout position needs to know which sequence positions hold image
content. The processor expands a single `<image>` placeholder into 64 real
image tokens, so they can be located by id:

```python
self.register_buffer(
    "image_positions",
    (tokens["input_ids"] == processor.tokenizer.convert_tokens_to_ids(
        "<image>")).unsqueeze(-1), persistent=False)
```

Why a second readout position at all is the subject of the next chapter, and
the answer turns out to be forced by how causal attention works.
