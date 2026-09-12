import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import gymnasium as gym
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
import pygame
from gym_robotics_custom import HeldSetpointWrapper
from controller import Controller

# Joystick button, not a keyboard key. Keyboard events need a pygame display
# surface to attach to, which this script never creates -- keys typed into
# the MuJoCo window (the one actually focused) would never reach
# pygame.event.get(). Joystick buttons don't have that problem: they're read
# directly off the device regardless of window focus, same as quit_pressed
# and abort_pressed already do in controller.py.
#
# Plus, on the same 8BitDo/Switch-mode pad controller.py's buttons were
# measured for (Minus = 9). Re-measure if it doesn't fire.
SAVE_BUTTON = 10
POSES_PATH = "start_poses.yaml"
DEFAULT_FREQUENCY = 1.0


def load_poses(path):
    try:
        with open(path) as f:
            return yaml.safe_load(f) or []
    except FileNotFoundError:
        return []


def save_poses(path, poses):
    with open(path, "w") as f:
        yaml.safe_dump(poses, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    env = gym.make("FrankaKitchen-v1", max_episode_steps=100000,
                    tasks_to_complete=["microwave"], render_mode='human')
    env = HeldSetpointWrapper(env)

    env.reset()

    controller = Controller()
    controller.reset()

    poses = load_poses(POSES_PATH)
    print(f"Loaded {len(poses)} existing poses from {POSES_PATH}")

    # A fresh file always gets the env's own start pose as entry 0, so
    # RandomStartWrapper never draws from a library that's missing the one
    # pose reset_model itself considers home -- no need to remember to
    # drive back and press Plus for it.
    if not poses:
        home_qpos = env.unwrapped.data.qpos[:9].copy().tolist()
        poses.append({"qpos": home_qpos, "frequency": DEFAULT_FREQUENCY})
        print(f"Seeded pose 1 (home, auto-captured): {home_qpos}")

    print("Drive the arm to a pose spanning the workspace. "
          "Plus saves it, Minus finishes.")

    running = True
    save_held = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if controller.quit_pressed():
            running = False
            break

        # Edge-triggered: only fire on the press, not every 50ms it's held.
        if controller.joystick.get_button(SAVE_BUTTON):
            if not save_held:
                qpos = env.unwrapped.data.qpos[:9].copy().tolist()
                poses.append({"qpos": qpos, "frequency": DEFAULT_FREQUENCY})
                print(f"Saved pose {len(poses)}: {qpos}")
            save_held = True
        else:
            save_held = False

        action = controller.get_action()
        if action is not None:
            env.step(action)

        time.sleep(0.05)

    env.close()

    save_poses(POSES_PATH, poses)
    print(f"Wrote {len(poses)} poses to {POSES_PATH}")
