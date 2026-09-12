"""Fit head variants on cached VLM outputs. Minutes, not hours.

The VLM is frozen, so its output for a given (frame, instruction) never changes.
Encode a sample once, then any number of readout and head ideas can be compared
without a training run. Held out by shard, because adjacent frames in an episode
are nearly the same picture.

  python scripts/probe.py            reuse the cache if present
  python scripts/probe.py --encode   rebuild it
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import frames
from agent import GRIPPER_WEIGHT
from model import Model
from tasks import task_index

CACHE = "checkpoints/probe_encodings.pt"
TASKS = ["microwave", "hinge_cabinet", "top_burner"]
SHARDS_PER_TASK, HELD_OUT_SHARDS = 26, 6
BATCH, STEPS, LR, SEEDS = 64, 3000, 1e-2, 5


def weighted_mse(pred, target):
    return (F.mse_loss(pred[:, :7], target[:, :7])
            + GRIPPER_WEIGHT * F.mse_loss(pred[:, 7:], target[:, 7:]))


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


def build_cache():
    model = Model(num_actions=9, name="probe").cuda()
    rng = np.random.default_rng(0)
    train_shards, test_shards = [], []
    for task in TASKS:
        picked = rng.permutation(
            sorted(glob.glob(f"dataset/{task}/*.npz")))[:SHARDS_PER_TASK]
        test_shards += list(picked[:HELD_OUT_SHARDS])
        train_shards += list(picked[HELD_OUT_SHARDS:])

    start = time.time()
    sets = {"train": encode(model, train_shards),
            "test": encode(model, test_shards)}
    torch.save(sets, CACHE)
    print(f"encoded {len(sets['train']['last'])} train + "
          f"{len(sets['test']['last'])} held-out steps in {time.time() - start:.0f}s")
    return sets


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


class Single(nn.Module):
    def __init__(self, dim=960):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 9))

    def forward(self, x):
        return self.net(x)


class Fused(nn.Module):
    def __init__(self, hidden=0):
        super().__init__()
        self.norm_last = nn.LayerNorm(960)
        self.norm_pooled = nn.LayerNorm(960)
        self.out = (nn.Linear(1920, 9) if not hidden else
                    nn.Sequential(nn.Linear(1920, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 9)))

    def forward(self, last, pooled):
        return self.out(torch.cat([self.norm_last(last),
                                   self.norm_pooled(pooled)], dim=1))


if __name__ == "__main__":
    if "--encode" in sys.argv or not os.path.exists(CACHE):
        sets = build_cache()
    else:
        sets = torch.load(CACHE)
        print(f"cached: {len(sets['train']['last'])} train + "
              f"{len(sets['test']['last'])} held-out steps")

    a_train, a_test = sets["train"]["actions"], sets["test"]["actions"]
    print(f"predict-the-mean, held out: "
          f"{weighted_mse(a_train.mean(0).expand_as(a_test), a_test):.4f}\n")
    print(f"{'head':<32}{'params':>10}{'mean':>9}{'best':>9}{'worst':>9}"
          f"   (held out, {SEEDS} seeds)")

    fit("last only", Single, ["last"], sets)
    fit("pooled only", lambda: Single(), ["pooled"], sets)
    fit("fused", Fused, ["last", "pooled"], sets)
    fit("fused -> 512 -> 9", lambda: Fused(hidden=512), ["last", "pooled"], sets)
