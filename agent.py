import os

# MuJoCo picks its GL backend at import time and defaults to GLFW, which needs
# an X display. Headless training servers have none, so fall back to EGL there.
# setdefault, so the environment can still override.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn.functional as F
import time
import datetime
import gymnasium as gym
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
from torch.optim.adam import Adam
from gym_robotics_custom import HeldSetpointWrapper, VLAObservationWrapper, ObsReshapeWrapper


from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset
from model import Model
from tasks import TASKS, TASK_DESCRIPTIONS, task_index

# The gripper dims (7 and 8, always identical) carry ~8x the variance of the
# average arm joint, so an unweighted mean over all 9 hands them 70% of the loss.
# 0.125 puts the one gripper dof and the seven arm joints on equal footing.
# Measured on all 56,005 steps: dims 7 and 8 are +/-1 in 100% of them.
GRIPPER_WEIGHT = 0.125

EVAL_TASKS = ["microwave", "hinge cabinet", "top burner"]
EVAL_ROLLOUTS = 3

class Agent:

    def __init__(self, eval=False, data_path="dataset", name='vla_network'):
        self.max_episode_steps = 400  # longest demo on file is 314; a policy still going at 400 has failed
        # SmolVLM2's vision tower wants 512. 448 is the largest even reduction of
        # the 896 archive that fits in RAM -- Dataset preallocates, so a step
        # costs 602 KiB here against 2.30 MiB at 896.
        self.image_size = 448
        self.native_image_size = 896
        # 56,005 steps on disk today. At 448 the arena is 36 GB; the old 100000
        # would ask for 60 GB.
        max_buffer_size = 60000
        # 1e-3, not the BC stack's 1e-4. The head reads a LayerNormed vector
        # whose informative component is a few percent of the whole; at 1e-4 the
        # ten-sample overfit still had loss 0.0075 after 400 steps.
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

        # Only the head. requires_grad_(False) on the VLM means Adam would carry
        # state for 507M parameters it can never move.
        self.optimizer = Adam(self.model.head.parameters(), learning_rate)

    def _make_env(self, task, render_mode):
        env = gym.make("FrankaKitchen-v1", max_episode_steps=self.max_episode_steps,
                       tasks_to_complete=[task], render_mode=render_mode)
        env = HeldSetpointWrapper(env)
        env = VLAObservationWrapper(env, image_size=self.native_image_size)
        return ObsReshapeWrapper(env, image_size=self.image_size)

    def train(self, epochs, batch_size):
        # Beekeeper injects BEEKEEPER_TENSORBOARD_DIR and serves whatever lands
        # there; locally it is unset and this stays runs/ as before.
        runs_dir = os.environ.get("BEEKEEPER_TENSORBOARD_DIR", "runs")
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        summary_writer = SummaryWriter(
            os.path.join(runs_dir, f"{stamp}_bs={batch_size}"))

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

            if(epoch % 10 == 0):
                summary_writer.add_scalar("train/loss", loss, epoch)
                summary_writer.add_scalar("train/arm", arm_loss, epoch)
                summary_writer.add_scalar("train/gripper", gripper_loss, epoch)

            if(epoch % 100 == 0):
                print(f"Epoch: {epoch} Loss: {loss.item()}")
                self.model.save_checkpoint()

            # Every 2500 rather than 1000: a rollout is ~400 VLM forwards at
            # batch 1, so an eval block costs minutes, and on a 100K run the old
            # cadence would spend hours of the budget evaluating.
            if(epoch and epoch % 2500 == 0):
                self.eval(epoch, summary_writer)

    def eval(self, epoch, summary_writer):
        rates = []
        for task in EVAL_TASKS:
            rate = sum(self.test(task) for _ in range(EVAL_ROLLOUTS)) / EVAL_ROLLOUTS
            summary_writer.add_scalar(f"eval/{task.replace(' ', '_')}", rate, epoch)
            print(f"  eval {task}: {rate:.0%}")
            rates.append(rate)
        # The headline. Per-task rates are 0, 0.33, 0.67 or 1 at three rollouts,
        # so the mean across tasks is the only eval number with any resolution.
        summary_writer.add_scalar("eval/mean", sum(rates) / len(rates), epoch)
        print(f"  eval mean: {sum(rates) / len(rates):.0%}")
        # The weights that produced these numbers. Without this the checkpoint on
        # disk is whichever eval happened to run last, lucky or not.
        torch.save(self.model.head.state_dict(),
                   f"{self.model.checkpoint_file}.e{epoch}")

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
                time.sleep(delay)
        self.model.train()

        env.close()
        return total_reward > 0
