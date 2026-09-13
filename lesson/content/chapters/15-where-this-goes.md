---
title: "What you have, and what's next"
part: "Part IV · Lessons"
chapter: 15
weight: 15
standfirst: "A working policy, a measured ceiling, and four levers in the order they become worth pulling."
---

## Where it landed

A frozen 507M-parameter vision-language model with a 21,129-parameter head,
trained on 56,005 demonstration steps, driving a Franka arm from one camera
frame and one English sentence.

| task | success | 95% CI |
|---|---|---|
| slide cabinet | 96% | [86.5, 98.9] |
| hinge cabinet | 68% | [54.2, 79.2] |
| top burner | 54% | [40.4, 67.0] |
| bottom burner | 8% | [3.2, 18.8] |
| kettle | 0% | [0.0, 7.1] |
| light switch | 0% | [0.0, 7.1] |
| microwave | ~2% | (in-loop, n≈120) |

The interesting part is no longer whether it works but the shape of the
failure. Three tasks performed, one marginal, three untouched. A uniformly
mediocre policy would suggest a capacity problem. This does not.

## What the ceiling looks like

Three pieces of evidence that the frozen representation, not the head, is the
binding constraint:

- A head with **47× more parameters** scores worse and is unstable enough that
  its worst seed lands on the predict-the-mean baseline.
- Held-out MSE bottoms out near **0.0905** against a 0.2132 baseline, so the
  readout explains roughly 58% of action variance, and no head architecture
  tried moves it.
- Checkpoints **trade tasks against each other** rather than improving
  together, which is what interference looks like when a small head holds
  several behaviors over a fixed representation.

## The levers, in order

### 1 · Unfreeze the top of the text stack

The first real test of the ceiling hypothesis. Match parameters by name rather
than hardcoding a module path, so a rename in `transformers` fails loudly
instead of silently training nothing:

<p class="filename">Filename: <strong>model.py</strong></p>

```python
# Matched against parameter names rather than hardcoding an attribute path
# (self.vlm.text_model.layers[...]) because that path is specific to the
# current Idefics3/SmolVLM class and would silently no-op if HF renames it.
_TEXT_LAYER_RE = re.compile(r"(?:text_model|language_model)\.layers\.(\d+)\.")


def _unfreeze_last_text_layers(vlm, n):
    """Set requires_grad on the last n text-decoder layers' parameters."""
    if n <= 0:
        return
    indices = {int(m.group(1)) for name, _ in vlm.named_parameters()
               if (m := _TEXT_LAYER_RE.search(name))}
    if not indices:
        raise RuntimeError(
            "UNFREEZE_LAST_N_LAYERS is set but no 'text_model.layers.N.' or "
            "'language_model.layers.N.' parameters were found on the VLM -- "
            "backbone naming has changed, update _TEXT_LAYER_RE.")
    if n > len(indices):
        raise RuntimeError(
            f"UNFREEZE_LAST_N_LAYERS={n} but the text tower only has "
            f"{len(indices)} layers.")
    keep = set(sorted(indices)[-n:])
    for name, p in vlm.named_parameters():
        m = _TEXT_LAYER_RE.search(name)
        if m and int(m.group(1)) in keep:
            p.requires_grad_(True)
```

Unfrozen weights need their own optimizer group at a tenth of the head's
learning rate. A head-sized rate on pretrained weights destroys them quickly:

<p class="filename">Filename: <strong>agent.py</strong></p>

```python
# Head always trains at learning_rate. Any unfrozen VLM layers train slower in
# their own group -- they're pretrained, so a head-sized LR would wreck them.
backbone_lr = float(os.environ.get("BACKBONE_LR", learning_rate * 0.1))
param_groups = [{"params": self.model.head.parameters(), "lr": learning_rate}]
vlm_params = self.model.trainable_vlm_parameters()
if vlm_params:
    param_groups.append({"params": vlm_params, "lr": backbone_lr})
    print(f"agent: training {sum(p.numel() for p in vlm_params):,} "
          f"backbone params at lr={backbone_lr}")
self.optimizer = Adam(param_groups)
```

Two layers is 19,664,640 parameters, roughly a thousand times the head. Note
that this also invalidates the probe cache from
[Chapter 12]({{< relref "chapters/12-the-fast-loop" >}}), and changes checkpoints from 85 KB to
37 MB.

### 2 · Diagnose the dead tasks

Three tasks at flat zero is not a gradual capability limit, it is something
categorical. Candidates: too few demonstrations for those tasks, an instruction
that does not distinguish itself in embedding space, or start-pose geometry
that puts the goal outside the camera frame. Cheap to investigate, and possibly
worth more than any architecture change.

### 3 · Action chunking, k = 8

The standard answer to compounding error, and the failure mode on the
long-horizon tasks looks like exactly that. This is where the live forward pass
from [Chapter 2]({{< relref "chapters/02-four-decisions" >}}) earns itself: build embeddings from
`input_ids`, concatenate k learned query vectors, extend the attention mask by
k, and pass `pixel_values` alongside with `input_ids=None`. Impossible with a
cached prefix.

### 4 · A proprioceptive state token

`Linear(9 → 960)` in the prefix. Deliberately left out of the first build:
with joint position wired to the head, the model can partially fit while the
visual prefix is garbage, and [the gate]({{< relref "chapters/11-the-gate" >}}) loses its ability to
tell you. It rides in on the machinery chunking already needs.

## The test nobody has run yet

Everything above is about control. The reason to put a *language* model in the
loop is generalization over instructions, and that has never been tested,
because an `nn.Embedding(7)` would pass every conditioning check in this guide.

The real test is a sentence the model never trained on. "Open the cabinet on
the left" instead of "Open the cabinet second from the left." If the arm still
moves correctly, the language model is doing something a lookup table cannot.
If it does not, this is an expensive seven-way classifier and the honest move
is to say so.

That is the first experiment the previous architecture could not have run, and
it is what the whole build was for.

---

<div class="finding">
<span class="note-label">If you take one thing</span>
The architecture in this guide worked on day one. What cost time was accepting
readings from instruments that could not have produced a different answer.
Build the measurement with the same care as the thing being measured, and
compute its variance before you trust it.
</div>
