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
[Chapter 7]({{< relref "chapters/07-the-readout" >}}) — four architectures,
five seeds each, twenty fits — is minutes of work. As training runs it would be
a day.

## Caching both readout streams

<p class="filename">Filename: <strong>scripts/probe.py</strong></p>

```python
CACHE = "checkpoints/probe_encodings.pt"
TASKS = ["microwave", "hinge_cabinet", "top_burner"]
SHARDS_PER_TASK, HELD_OUT_SHARDS = 26, 6
BATCH, STEPS, LR, SEEDS = 64, 3000, 1e-2, 5


def encode(model, shards):
    images, actions, tasks = [], [], []
    for path in shards:
        data = np.load(path)
        images.append(frames.resize(data["camera_scene"], 448))
        actions.append(data["action"])
        tasks.append(np.full(len(data["action"]),
                             task_index(str(data["task_description"]))[0]))
    images, tasks = np.concatenate(images), np.concatenate(tasks)

    last, pooled = [], []
    with torch.no_grad():
        for i in range(0, len(images), BATCH):
            task = torch.tensor(tasks[i:i + BATCH]).to(model.device)
            h = model.vlm(input_ids=model.prompt_ids[task],
                          attention_mask=model.prompt_mask[task],
                          pixel_values=model.preprocess(images[i:i + BATCH])
                          ).last_hidden_state.float()
            last.append(h[torch.arange(len(task), device=model.device),
                          model.prompt_end[task]])
            mask = model.image_positions[task]
            pooled.append((h * mask).sum(1) / mask.sum(1))

    return {"last": torch.cat(last), "pooled": torch.cat(pooled),
            "actions": torch.tensor(np.concatenate(actions),
                                    dtype=torch.float32).to(model.device)}
```

Both readout streams are stored, so a variant can use either or both without
re-encoding. This is the same gather-and-pool from
[Chapter 7]({{< relref "chapters/07-the-readout" >}}), lifted out of `forward`
and run once.

<div class="note">
<span class="note-label">Scope, and its expiry</span>
This is the cache <a href="{{< relref "chapters/02-four-decisions" >}}">Chapter 2</a>
refused to train on, reintroduced as a diagnostic. It is valid exactly as long
as the VLM is frozen and the prompt is unchanged. Unfreeze one layer and every
vector in that file is stale.
</div>

## Hold out by shard, not by step

The 56,005 steps come from 582 episodes. Consecutive steps inside an episode
are nearly identical. Split randomly by step and you put near-duplicates of
training frames in the validation set, making validation loss an optimistic
fiction.

So the split happens at the level of whole files, before any encoding:

<p class="filename">Filename: <strong>scripts/probe.py</strong></p>

```python
def build_cache():
    model = Model(num_actions=9, name="probe").cuda()
    rng = np.random.default_rng(0)
    train_shards, test_shards = [], []
    for task in TASKS:
        picked = rng.permutation(
            sorted(glob.glob(f"dataset/{task}/*.npz")))[:SHARDS_PER_TASK]
        test_shards += list(picked[:HELD_OUT_SHARDS])
        train_shards += list(picked[HELD_OUT_SHARDS:])

    sets = {"train": encode(model, train_shards),
            "test": encode(model, test_shards)}
    torch.save(sets, CACHE)
    return sets
```

A whole episode is either training or held out. The seeded `default_rng(0)`
keeps the split identical across runs, so two variants measured a week apart
are still comparable.

## One seed is not a measurement

<p class="filename">Filename: <strong>scripts/probe.py</strong></p>

```python
def fit(label, build, inputs, sets, lr=LR, steps=STEPS, seeds=SEEDS):
    """Same data, several seeds. One seed swings by more than the effects here."""
    a_train, a_test = sets["train"]["actions"], sets["test"]["actions"]
    train_in = [sets["train"][k] for k in inputs]
    test_in = [sets["test"][k] for k in inputs]
    results = []

    for seed in range(seeds):
        torch.manual_seed(seed)
        head = build().cuda()
        optimizer = torch.optim.Adam(head.parameters(), lr)
        for _ in range(steps):
            index = torch.randint(len(a_train), (256,), device=a_train.device)
            loss = weighted_mse(head(*[t[index] for t in train_in]), a_train[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            results.append(weighted_mse(head(*test_in), a_test).item())

    params = sum(p.numel() for p in head.parameters())
    print(f"  {label:<30}{params:>10,}{np.mean(results):>9.4f}"
          f"{min(results):>9.4f}{max(results):>9.4f}")
```

`inputs` is a list of keys into the cache, so one function fits every variant:

```python
fit("last only", Single, ["last"], sets)
fit("pooled only", lambda: Single(), ["pooled"], sets)
fit("fused", Fused, ["last", "pooled"], sets)
fit("fused -> 512 -> 9", lambda: Fused(hidden=512), ["last", "pooled"], sets)
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

Report the baseline alongside, or the numbers have no scale:

```python
print(f"predict-the-mean, held out: "
      f"{weighted_mse(a_train.mean(0).expand_as(a_test), a_test):.4f}")
```

The shipped configuration survived the correction. Fused was best at 0.0905
mean, with the tightest spread of any variant. But it survived as a measurement
rather than as a lucky draw.

This same error is about to repeat at a much larger scale, in a place where it
costs a whole night.
