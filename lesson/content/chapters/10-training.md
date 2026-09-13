---
title: "The training loop"
part: "Part II · The pieces"
chapter: 10
weight: 10
standfirst: "Ordinary behavior cloning, one weighted loss term, and numbers for budgeting a run."
---

Sample a batch, run it forward, compare against what the human did, step the
optimizer. There is nothing exotic in this chapter, which is deliberate: the
novel part of this build is the prefix, and we want a training loop boring
enough that when something goes wrong we know it isn't here.

## Setting up

<p class="listing">Listing 10.1 <em>Agent construction</em></p>
<p class="filename">Filename: <strong>agent.py</strong></p>

```python
class Agent:

    def __init__(self, eval=False, data_path="dataset", name='vla_network'):
        self.max_episode_steps = 400   # longest demo on file is 314
        # 448 is the largest even reduction of 896 that fits: Dataset
        # preallocates, so 56,005 steps cost 36 GB here and 129 GB at 896.
        self.image_size = 448
        self.native_image_size = 896
        max_buffer_size = 60000
        learning_rate = 0.001

        env = self._make_env(EVAL_TASKS[0], render_mode='rgb_array')
        obs, _ = env.reset()

        self.dataset = Dataset(max_size=max_buffer_size,
                               image_size=self.image_size,
                               n_actions=env.action_space.shape[0],
                               n_joints=9)
        if not eval:
            self.dataset.load_data(path=data_path)

        num_actions = env.action_space.shape[0]
        env.close()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = Model(num_actions=num_actions, name=name).to(self.device)
        self.optimizer = Adam(self.model.head.parameters(), lr=learning_rate)
```

Two details deserve a moment. `num_actions` comes from the environment rather
than from a constant, so the head's output width can never quietly disagree with
what the environment will accept.

And the optimizer is handed `self.model.head.parameters()`, not
`self.model.parameters()`. The backbone has `requires_grad=False` so its
gradients would be `None` either way, but handing Adam 460 million frozen
tensors makes it allocate optimizer state for every one of them.

## The loop

<p class="listing">Listing 10.2 <em>Behavior cloning, with the gripper weighted down</em></p>
<p class="filename">Filename: <strong>agent.py</strong></p>

```python
# Gripper dims are +/-1 in every step and carry ~8x an arm joint's variance.
GRIPPER_WEIGHT = 0.125


def train(self, epochs, batch_size):
    summary_writer = SummaryWriter(...)

    for epoch in range(epochs):
        states, actions, _, _, tasks = self.dataset.sample_batch(batch_size)

        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        tasks   = torch.tensor(tasks).to(self.device)

        pred_actions = self.model(states['camera_scene'], tasks)

        arm_loss     = F.mse_loss(actions[:, :7], pred_actions[:, :7])
        gripper_loss = F.mse_loss(actions[:, 7:], pred_actions[:, 7:])

        loss = arm_loss + GRIPPER_WEIGHT * gripper_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if epoch % 100 == 0:
            self.model.save_checkpoint()

        if epoch and epoch % 2500 == 0:
            self.eval(epoch, summary_writer)
```

The split loss comes straight from the statistics we printed in Chapter 4.
Dimensions 7 and 8 are binary and carry about eight times an arm joint's
variance, so `GRIPPER_WEIGHT = 0.125` puts them back on comparable footing.
Log the two halves separately as well as the total, or you won't be able to tell
which one is moving.

Here's what the first few thousand epochs look like:

<div class="output"><p class="output-label">A real run, printing every 100 epochs</p>

```text
Loaded 56005 steps from /data/datasets/farama-kitchen-bc/dataset in 68.7s
device: NVIDIA GeForce RTX 3090
Epoch: 0 Loss: 0.4890671968460083
Epoch: 100 Loss: 0.13493306934833527
Epoch: 200 Loss: 0.1449355036020279
Epoch: 300 Loss: 0.10025772452354431
Epoch: 400 Loss: 0.11838166415691376
Epoch: 500 Loss: 0.09118559956550598
```
</div>

It drops fast and then starts bouncing around in a band. That band is normal here
and it stays roughly 0.05 to 0.11 for the rest of the run. Do not read anything
into small movements in it; Chapter 13 is a long argument about why.

## The rollout

Evaluation runs the policy in the environment, and it's the only code that
exercises the environment path at all, which makes it worth more than its line
count suggests.

<p class="listing">Listing 10.3 <em>One rollout</em></p>
<p class="filename">Filename: <strong>agent.py</strong></p>

```python
def test(self, task, render_mode="rgb_array", delay=0):
    env = self._make_env(task, render_mode)
    task_id = torch.tensor(task_index(TASKS[task])).to(self.device)
    obs, _ = env.reset()
    done = trunc = False
    total_reward = 0.0

    self.model.eval()
    with torch.no_grad():
        while not (done or trunc):
            action = self.model(obs['camera_scene'], task_id)
            obs, reward, done, trunc, _ = env.step(action.cpu().numpy().squeeze())
            total_reward += reward
    self.model.train()

    env.close()
    return total_reward > 0
```

Success is any reward at all, because Franka Kitchen only pays out on task
completion. The `squeeze()` drops the batch dimension the model always adds.

## Numbers for budgeting

<dl class="stats">
  <div><dt>Batch</dt><dd>64</dd></div>
  <div><dt>Head LR</dt><dd>1e-3</dd></div>
  <div><dt>Throughput</dt><dd>2.55/s</dd></div>
  <div><dt>VRAM</dt><dd>2.9 GB</dd></div>
</dl>

At 2.55 epochs per second on an RTX 3090, a hundred thousand epochs is about
eleven hours of training plus roughly a hundred minutes of evaluation. Loading
the dataset takes another 69 seconds at startup.

<div class="checkpoint">
<span class="note-label">Don't start a long run yet</span>
Everything compiles and the loss goes down, and neither of those is evidence
that it works. Part III is three cheap tests that tell you whether what you just
built is learning, in increasing order of cost, and the first one takes under a
minute.
</div>

Next, we'll build that first test, which asks three questions in an order where
each one only makes sense if the previous answer was yes.
