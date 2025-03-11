#!/usr/bin/python3

from marp_ai_gym import *
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import datetime
import os

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq  # Save frequency in timesteps
        self.save_path = save_path    # Path to save the model

    def _on_step(self) -> bool:
        # Check if the current step is a multiple of the save frequency
        if self.n_calls % self.save_freq == 0:
            # Save the model
            self.model.save(f"{self.save_path}/autosave_step_{self.num_timesteps}")
            if self.verbose > 0:
                print(f"Model saved at timestep {self.num_timesteps}")
        return True
    
def ppo_train():
    def make_env():
        def _init():
            env = MarpAIGym(render_flag=False)
            env = Monitor(env)
            return env
        return _init 
    num_envs = 1
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=1024, batch_size=64, n_epochs=20, verbose=1)
    save_path = os.path.join("saves", "autosave_sb_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    callback = SaveModelCallback(save_freq=100000, save_path=save_path, verbose=1)
    model.learn(total_timesteps=1000000, callback=callback)
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
