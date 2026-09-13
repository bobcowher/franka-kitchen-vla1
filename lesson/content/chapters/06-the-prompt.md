---
title: "The prompt"
part: "Part II · The pieces"
chapter: 6
weight: 6
standfirst: "Seven instructions, tokenized once, and a padding bug that never raises."
---

There are seven tasks and therefore seven instructions, and they never change.
That last fact is worth more than it sounds, because it means the entire text
side of our input can be computed once at construction and then indexed.

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

A task's identity through the whole system is its position in this dictionary.
Reorder it and every existing checkpoint maps "open the microwave" onto weights
learned for a different task, with nothing anywhere raising.

## Tokenizing all seven at once

Let's build the prompts. The chat template wants a message with an image
placeholder and the instruction text, and the processor needs some image to
expand that placeholder into real token positions, so we hand it a black square
and throw the pixels away.

<p class="listing">Listing 6.1 <em>Building and tokenizing the seven prompts</em></p>
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

This is not the prefix caching we ruled out in Chapter 2. We're storing token
ids, not model outputs, so unfreezing the backbone later invalidates none of it.

## The padding problem

Let's look at what the tokenizer returned:

```python
print("prompt_ids shape  ", tuple(tokens["input_ids"].shape))
print("real lengths      ", tokens["attention_mask"].sum(1).tolist())
```

<div class="output"><p class="output-label">This prints</p>

```text
prompt_ids shape   (7, 84)
real lengths       [81, 83, 81, 79, 84, 84, 82]
```
</div>

Every row is 84 long, but only two of them contain 84 real tokens. "Open the
microwave door" is 79 tokens and "Turn the oven knob for the bottom left burner"
is 84, so five of the seven rows have been padded on the right.

<div class="trap">
<span class="note-label">Trap</span>
<p>Our head is going to read the last position of the sequence. If we reach for
<code>last_hidden_state[:, -1]</code> we will read a <strong>pad token</strong>
for five of the seven tasks. Nothing raises, nothing warns, and the loss curve
looks entirely normal. The model simply learns worse for those five tasks.</p>
<p>Left-padding looks like the obvious fix and is a different bug.
<em>RoPE</em>, the rotary position embedding this model uses, encodes each
token's place in the sequence from its absolute index. Padding on the left
shifts every index, so the model would see the image at positions it was never
pretrained to see it at.</p>
</div>

The answer is to right-pad and then *gather* each row's own last real token,
which the attention mask already tells us how to find:

<p class="listing">Listing 6.2 <em>Registering the prompts and the gather index</em></p>
<p class="filename">Filename: <strong>model.py</strong></p>

```python
self.register_buffer("prompt_ids", tokens["input_ids"], persistent=False)
self.register_buffer("prompt_mask", tokens["attention_mask"], persistent=False)
# Right-padded, so the last real token is at mask.sum() - 1. Left
# padding would shift every RoPE position.
self.register_buffer("prompt_end", tokens["attention_mask"].sum(1) - 1,
                     persistent=False)
```

<div class="output"><p class="output-label">Printing <code>prompt_end</code> gives</p>

```text
[80, 82, 80, 78, 83, 83, 81]
```
</div>

Because attention is *causal*, meaning each position attends only to positions
before it and never after, a real token cannot see a pad token that follows it.
The hidden state we gather at index 78 for the microwave prompt is therefore
bit-identical to what we would get by running that 79-token sequence with no
padding at all. No masking subtleties and no position shift.

`persistent=False` keeps these three tensors out of the saved state dict, since
they're derived from the tokenizer rather than learned.

## Finding the image tokens

The readout in the next chapter reads the image positions too. The processor
expands our single `<image>` placeholder into a run of real image tokens, all
sharing one id:

<p class="filename">Filename: <strong>model.py</strong></p>

```python
self.register_buffer(
    "image_positions",
    (tokens["input_ids"] == processor.tokenizer.convert_tokens_to_ids(
        "<image>")).unsqueeze(-1), persistent=False)
```

<div class="output"><p class="output-label">The image token id, and the count per prompt</p>

```text
image token id           : 49190
image tokens per prompt  : [64, 64, 64, 64, 64, 64, 64]
```
</div>

Sixty-four image tokens in every prompt, which makes sense: the vision tower
takes a 512-pixel image at patch 16, giving a 32×32 grid that gets pooled down to
8×8. The `unsqueeze(-1)` adds a trailing dimension so the mask will broadcast
over the 960 features when we use it.

<div class="checkpoint">
<span class="note-label">Check before you continue</span>
Print <code>prompt_mask.sum(1)</code> and confirm you get seven <em>different</em>
numbers between 79 and 84. Seven identical numbers means padding isn't being
applied and you're about to debug something much harder later. Then confirm
<code>image_positions.sum(1)</code> is 64 for every task.
</div>

<div class="exercise">
<h4>Exercise 6.1 &nbsp;See the bug you just avoided</h4>
<p>Gather the hidden state two ways for a batch containing several different
tasks: once at <code>prompt_end</code>, and once at <code>-1</code>. Print the
maximum absolute difference per row.</p>
<p>Rows for the two longest instructions will agree exactly. The other five
won't, and the size of that difference is what a silent bug looks like when you
finally measure it.</p>
</div>

Next: which positions the head reads, and why the prompt ordering forces the
answer.
