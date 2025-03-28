import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from masked_fcnet_model import MaskedFCNet 
from marp_ai_gym_rllib import *
import os
import matplotlib.pyplot as plt

# Register the custom model
ModelCatalog.register_custom_model("masked_fcnet", MaskedFCNet)

# === User Inputs Test Parameters==
max_level = 9
episodes_per_level = 50

# Register the custom environment
env_name = "marp_ai_env"
register_env(env_name, lambda config: MarpAIGym(config, render_flag=False))

# Define the checkpoint path (update this to your actual checkpoint location)
pwd = os.getcwd()
checkpoint_path = os.path.join(pwd, "models/PPO_2025-03-28_23-25-10/PPO_marp_ai_env_d7500_00000_0_2025-03-28_23-25-12/checkpoint_000008")

# Load the trained model
config = (
    PPOConfig()
    .environment(env=env_name)
    .framework("torch")
    .rollouts(num_rollout_workers=0)  # No workers needed for testing
)

algo = PPO.from_checkpoint(checkpoint_path)

# === Tracking Data ===
solve_rates = []
collision_rates = []
average_rewards = []

print(f"\n=== Testing PPO Model across Levels 0 to {max_level} ===\n")
for level in range(max_level + 1):
    env = MarpAIGym(render_flag=True)
    env.level = level
    env.selected_level = level

    solved_cases = 0
    collision_cases = 0
    total_rewards = []

    for episode in range(episodes_per_level):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            episode_reward += reward

            #Success: Both AMRs reach goal simultaneously
            if env.amr1_pose == env.amr1_dest and env.amr2_pose == env.amr2_dest:
                solved_cases += 1
                break

            #Swap = treated as collision
            if env.amr1_last_pose == env.amr2_pose and env.amr1_pose == env.amr2_last_pose:
                collision_cases += 1
                break

            #Collision
            if env.amr1_pose == env.amr2_pose:
                collision_cases += 1
                break

        total_rewards.append(episode_reward)

    # === Metrics for this level ===
    solve_rate = (solved_cases / episodes_per_level) * 100
    collision_rate = (collision_cases / episodes_per_level) * 100
    avg_reward = sum(total_rewards) / len(total_rewards)

    solve_rates.append(solve_rate)
    collision_rates.append(collision_rate)
    average_rewards.append(avg_reward)

    print(f"Level {level} | Solved: {solve_rate:.2f}% | Collisions: {collision_rate:.2f}% | Avg Reward: {avg_reward:.2f}")

# === Summary Table ===
print("\n=== Summary Scores ===")
print("{:<8} {:<15} {:<18} {:<15}".format("Level", "Solve Rate (%)", "Collision Rate (%)", "Avg Reward"))
for lvl in range(max_level + 1):
    print("{:<8} {:<15.2f} {:<18.2f} {:<15.2f}".format(
        lvl, solve_rates[lvl], collision_rates[lvl], average_rewards[lvl]
    ))

# === Plotting Results ===
levels = list(range(max_level + 1))

plt.figure(figsize=(14, 4))
plt.suptitle("Model Performance", fontsize=16)

plt.subplot(1, 3, 1)
plt.plot(levels, solve_rates, marker='o')
plt.title("Solve Rate (%)")
plt.xlabel("Level")
plt.ylabel("Solve Rate")

plt.subplot(1, 3, 2)
plt.plot(levels, collision_rates, marker='o', color='orange')
plt.title("Collision Rate (%)")
plt.xlabel("Level")
plt.ylabel("Collision Rate")

plt.subplot(1, 3, 3)
plt.plot(levels, average_rewards, marker='o', color='green')
plt.title("Average Reward")
plt.xlabel("Level")
plt.ylabel("Reward")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
plt.show()