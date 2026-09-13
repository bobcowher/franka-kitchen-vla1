---
title: "The fast loop"
part: "Part III · Knowing whether it works"
chapter: 12
weight: 12
standfirst: "The frozen model's output never changes. That turns a three-hour experiment into a sixty-second one."
---

Because the VLM is frozen, its output for a given (frame, instruction) pair is
a constant. Encode a sample of the dataset once, keep the vectors, and every
question about the *head* can be answered without running the VLM again.

```text
python scripts/probe.py --encode    # ~60s, writes checkpoints/probe_encodings.pt
python scripts/probe.py             # fits head variants in seconds
```

Anything that changes only the head, the readout, the loss weighting or the
normalization goes here first. The table in
[Chapter 7]({{< relref "chapters/07-the-readout" >}}) — four architectures, five seeds each, twenty
fits — is minutes of work. As training runs it would be a day.

## Caching the encodings

<p class="filename">Filename: <strong>scripts/probe.py</strong></p>

```python
def encode(agent, n, path):
    """Run the frozen VLM once and keep both readout streams."""
    last, pooled, actions, shards = [], [], [], []
    with torch.no_grad():
        for batch in batches(agent.dataset, n):
            h = agent.model.vlm(
                input_ids=agent.model.prompt_ids[batch.tasks],
                attention_mask=agent.model.prompt_mask[batch.tasks],
                pixel_values=agent.model.preprocess(batch.frames),
            ).last_hidden_state.float()

            mask = agent.model.image_positions[batch.tasks]
            last.append(h[torch.arange(len(batch.tasks)),
                          agent.model.prompt_end[batch.tasks]].cpu())
            pooled.append(((h * mask).sum(1) / mask.sum(1)).cpu())
            actions.append(batch.actions)
            shards.append(batch.shards)

    torch.save({"last": torch.cat(last), "pooled": torch.cat(pooled),
                "actions": torch.cat(actions), "shards": torch.cat(shards)}, path)
```

Both streams are stored, so a variant can use either or both without
re-encoding.

<div class="note">
<span class="note-label">Scope, and its expiry</span>
This is the cache <a href="{{< relref "chapters/02-four-decisions" >}}">Chapter 2</a> refused to
train on, reintroduced as a diagnostic. It is valid exactly as long as the VLM
is frozen and the prompt is unchanged. Unfreeze one layer and every vector in
that file is stale.
</div>

## Hold out by shard, not by step

The 56,005 steps come from 582 episodes. Consecutive steps inside an episode
are nearly identical. Split randomly by step and you put near-duplicates of
training frames in the validation set, making validation loss an optimistic
fiction.

```python
held_out = torch.isin(data["shards"], eval_shards)
train_idx, test_idx = (~held_out).nonzero(), held_out.nonzero()
```

A whole episode is either training or held out.

## One seed is not a measurement

<p class="filename">Filename: <strong>scripts/probe.py</strong></p>

```python
def fit(label, build, inputs, sets, seeds=5, steps=2000):
    scores = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        head = build().cuda()
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        for _ in range(steps):
            loss = weighted_mse(head(*inputs.train), sets.train_actions)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            scores.append(weighted_mse(head(*inputs.test), sets.test_actions).item())
    print(f"{label:<28}{np.mean(scores):.4f}  {min(scores):.4f}  {max(scores):.4f}")
```

An earlier version fit each variant once. The fused head read **0.0934** in one
run and **0.1197** in another, on identical cached data, differing only in
initialization. A 28% swing, larger than the differences between the
architectures being compared. A confident claim had already gone into a commit
message on the strength of the first number.

<div class="finding">
<span class="note-label">Rule</span>
Repeat every fit across five seeds and report mean, best and worst. If the
spread between seeds is wider than the gap between your variants, you have not
measured anything — and you cannot know that from a single number.
</div>

The shipped configuration survived the correction. Fused was best at 0.0905
mean, with the tightest spread of any variant. But it survived as a measurement
rather than as a lucky draw.

This same error is about to repeat at a much larger scale, in a place where it
costs a whole night.
