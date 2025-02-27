#!/usr/bin/python3

from marp_ai_gym import *
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def ppo_train():
    def make_env():
        def _init():
            env = MarpAIGym(render_flag=False)
            env = Monitor(env)
            return env
        return _init 
    num_envs = 1
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    policy_kwargs = dict(net_arch=[128, 128, 128, 128])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=1024, batch_size=64, n_epochs=20, verbose=1)
    model.learn(total_timesteps=1000000) 
    model.save("ppo_marp_ai_model")

def dqn_train():
    # this algo isnt performing well, need to further tune hyperparameters
    env = MarpAIGym()
    env = DummyVecEnv([lambda: env])
    model = DQN("MlpPolicy", env, verbose=1, learning_starts=1000, buffer_size=10000, batch_size=64, gamma=0.99)
    model.learn(total_timesteps=1000000)
    model.save("dqn_marp_ai_model")

if __name__ == "__main__":
    ppo_train()
