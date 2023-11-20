# -*- coding: utf-8 -*-

import numpy as np
import gym
from stable_baselines3 import PPO,A2C
import time
from stable_baselines3.common.evaluation import evaluate_policy
from quanser_robots import GentlyTerminating

if __name__ == '__main__':
    env = gym.make('Rotpen-500-v0')
    model = PPO.load("rotpen_PPO")
    print('Model loaded successfully!')
    time.sleep(1)
    obs = env.reset()
    for i in range(25000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action*5)
        env.render()
        if i%250==0:
            print(i)
        if done:
            print('Done')
            obs = env.reset()
    env.close()