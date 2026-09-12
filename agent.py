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

# Gripper dims are +/-1 in every step and carry ~8x an arm joint's variance.
GRIPPER_WEIGHT = 0.125

EVAL_TASKS = ["microwave", "hinge cabinet", "top burner"]
EVAL_ROLLOUTS = 3


class Agent:

    def __init__(self, eval=False, data_path="dataset", name='vla_network'):
        self.max_episode_steps = 400  # longest demo on file is 314; a policy still going at 400 has failed
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
        if self.device != 'cpu':
            print(f"device: {torch.cuda.get_device_name(0)}")

        self.model = Model(num_actions=num_actions, name=name).to(self.device)

        # Head always trains at learning_rate. Any unfrozen VLM layers
        # (model.py's UNFREEZE_LAST_N_LAYERS) train slower in their own group --
        # they're pretrained, so a head-sized LR would wreck them fast.
        backbone_lr = float(os.environ.get("BACKBONE_LR", learning_rate * 0.1))
        param_groups = [{"params": self.model.head.parameters(), "lr": learning_rate}]
        vlm_params = self.model.trainable_vlm_parameters()
        if vlm_params:
            param_groups.append({"params": vlm_params, "lr": backbone_lr})
            print(f"agent: training {sum(p.numel() for p in vlm_params):,} "
                  f"backbone params at lr={backbone_lr}")
        self.optimizer = Adam(param_groups)

    def _make_env(self, task, render_mode):
        env = gym.make("FrankaKitchen-v1", max_episode_steps=self.max_episode_steps,
                       tasks_to_complete=[task], render_mode=render_mode)
        env = HeldSetpointWrapper(env)
        env = VLAObservationWrapper(env, image_size=self.native_image_size)
        return ObsReshapeWrapper(env, image_size=self.image_size)

    def train(self, epochs, batch_size):
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

            if(epoch and epoch % 2500 == 0):
                self.eval(epoch, summary_writer)

    def eval(self, epoch, summary_writer):
        rates = []
        for task in EVAL_TASKS:
            rate = sum(self.test(task) for _ in range(EVAL_ROLLOUTS)) / EVAL_ROLLOUTS
            summary_writer.add_scalar(f"eval/{task.replace(' ', '_')}", rate, epoch)
            print(f"  eval {task}: {rate:.0%}")
            rates.append(rate)
        # Per-task rates are 0, 0.33, 0.67 or 1, so the mean is the signal.
        summary_writer.add_scalar("eval/mean", sum(rates) / len(rates), epoch)
        print(f"  eval mean: {sum(rates) / len(rates):.0%}")
        # Otherwise the checkpoint on disk is whichever eval ran last.
        torch.save(self.model.trainable_state_dict(),
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
