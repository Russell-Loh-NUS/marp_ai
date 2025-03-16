import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from marp_ai_gym_rllib import *
import os

# Register the custom environment
env_name = "marp_ai_env"
register_env(env_name, lambda config: MarpAIGym(config, render_flag=False))

# Define the checkpoint path (update this to your actual checkpoint location)
pwd = os.getcwd()
checkpoint_path = os.path.join(pwd, "/PPO_marp_ai_env_1b59d_00000_0_2025-03-16_12-42-46/checkpoint_000014")

# Load the trained model
config = (
    PPOConfig()
    .environment(env=env_name)
    .framework("torch")
    .rollouts(num_rollout_workers=0)  # No workers needed for testing
)

algo = PPO.from_checkpoint(checkpoint_path)

# Create the environment for testing
env = MarpAIGym(render_flag=True)

while True:
    # Run a test episode
    obs, _ = env.reset()
    done = False

    while not done:
        action = algo.compute_single_action(obs)  # Compute action for single agent
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"obs: {obs}")
        # input("Press Enter to continue...")

env.close()