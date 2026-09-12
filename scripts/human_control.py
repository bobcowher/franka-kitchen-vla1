import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
from dataset import DatasetShard
from tasks import TASKS, task_from_argv
import frames
# from agent import Agent
from gym_robotics_custom import HeldSetpointWrapper, VLAObservationWrapper, RandomStartWrapper
import pygame
from controller import Controller


if __name__ == '__main__':

    max_episode_steps=500 # max episode steps
    # Square camera frame, archived at full size; training reduces it in
    # load_data and a rollout reduces it with ObsReshapeWrapper. Taken from
    # frames.py rather than typed here, so the collector cannot drift from the
    # sizes the loader will accept. Costs ~509 KiB/step on disk.
    image_size = frames.RENDER_SIZE
    env_name = "FrankaKitchen-v1"

    task = task_from_argv(sys.argv)
    task_description = TASKS[task]
    task_no_spaces = task.replace(" ", "_")



    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task], render_mode='human')

    env = HeldSetpointWrapper(env)
    env = RandomStartWrapper(env, poses_path="start_poses.yaml")
    env = VLAObservationWrapper(env, image_size=image_size)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    controller = Controller()

    running = True

    while running: # Minus on the pad ends the session
        episode_steps = 0
        done = False
        state, info = env.reset()
        controller.reset()

        memory = DatasetShard(max_size=max_episode_steps,
                              image_size=image_size,
                              n_actions=env.action_space.shape[0],
                              task_name=task_no_spaces,
                              task_description=task_description)

        reward = 0

        while running and not done and episode_steps < max_episode_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                # Check if CTRL+H is pressed
                    if event.key == pygame.K_h and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        # Trigger the key event in MuJoCo
                        env.render()  # Ensure the environment handles the key event

            if controller.quit_pressed():
                print("Quitting. This episode is not saved.")
                running = False
                break

            if controller.abort_pressed():
                print("Aborted. Discarding episode.")
                controller.wait_for_release()
                break

            action = controller.get_action()
            if(action is not None):
                next_state, reward, done, _, _ = env.step(action)
                memory.store_transition(state, action, reward, done)
                print(f"Episode step: {episode_steps} Reward: , {reward} Total steps in memory: {memory.mem_ctr}")
                state = next_state
                episode_steps += 1
            time.sleep(0.05)
        

        if running and reward > 0:
            if controller.keep_demo():
                memory.save_to_csv()
                directory = f'dataset/{task_no_spaces}'
                shards = len([f for f in os.listdir(directory) if f.endswith('.npz')])
                print(f"{shards} demos recorded for {task}.")
            else:
                print("Discarded.")

    env.close()



