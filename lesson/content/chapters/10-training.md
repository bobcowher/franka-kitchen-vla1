---
title: "The training loop"
part: "Part II · The pieces"
chapter: 10
weight: 10
standfirst: "Ordinary behavior cloning, one weighted loss term, and numbers for budgeting a run."
---

Sample a batch, forward, mean squared error against the demonstrated action,
step Adam on the head. Nothing exotic.

## Setting up

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

`num_actions` comes from the environment rather than a constant, so the head
width can never disagree with what the environment accepts.

The optimizer is given `self.model.head.parameters()`, not
`self.model.parameters()`. The VLM has `requires_grad=False` so its gradients
would be `None` either way, but handing 507M frozen tensors to Adam makes it
allocate optimizer state for them.

## The loop

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

The split loss comes straight from the statistics in
[Chapter 4]({{< relref "chapters/04-the-dataset" >}}). The gripper dimensions are binary and carry
about 8× an arm joint's variance; unweighted they dominate the gradient and the
model optimizes an easy binary decision at the expense of positioning the arm.
Log `arm_loss` and `gripper_loss` separately as well as the total, or you
cannot tell which half is moving.

## The rollout

Evaluation runs the policy in the environment. This is the only code that
exercises the environment path, so it catches an entire class of bug the
training loop cannot.

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

Success is any reward at all, since Franka Kitchen only pays out on task
completion. The `squeeze()` drops the batch dimension the model always adds.

## Numbers for budgeting

<dl class="stats">
  <div><dt>Batch</dt><dd>64</dd></div>
  <div><dt>Head LR</dt><dd>1e-3</dd></div>
  <div><dt>Throughput</dt><dd>2.55/s</dd></div>
  <div><dt>VRAM</dt><dd>2.9 GB</dd></div>
</dl>

2.55 epochs per second on an RTX 3090, so 100,000 epochs is about 11 hours of
training plus roughly 100 minutes of in-loop evaluation.

<div class="checkpoint">
<span class="note-label">Do not start a long run yet</span>
Everything compiles and the loss goes down. That is not evidence it works.
Part III is three cheap tests that tell you whether the thing you just built is
learning, in ascending order of cost, and the first one takes under a minute.
</div>
