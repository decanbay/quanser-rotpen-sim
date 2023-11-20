"""
The minimal program that shows the basic control loop on the simulated swing-up.
"""

import gym
from quanser_robots import GentlyTerminating
from quanser_robots.qube import SwingUpCtrl


# ctrl = SwingUpCtrl()
# obs = env.reset()
# done = False
# while not done:
#     env.render()
#     act = ctrl(obs)
#     obs, _, done, _ = env.step(act)

# env.close()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 00:37:27 2021

@author: deniz
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from typing import Callable


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = gym.make('Qube-100-v0')
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init
# env = GentlyTerminating(gym.make('Qube-100-v0'))


def test_env(env, model, plot=True):
    '''
    Parameters
    ----------
    env : gym environment
    model : Model
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''
    d_t = 1/env.timing.dt_ctrl
    duration = 10
    t = int(duration/d_t)
    obsrvs = []
    rews = []
    obs = env.reset()
    actions = []
    s = time.time()
    for i in range(t):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rews.append(reward)
        obsrvs.append(obs)
        actions.append(action)
        env.render()
    e = time.time() - s
    print(e)
    t = np.arange(0, duration, step=d_t)
    if plot:
        _, ax = plt.subplots(4)
        theta = [obs[0] for obs in obsrvs]
        alpha = [obs[1] for obs in obsrvs]
        ax[0].plot(t, theta, 'b')
        ax[1].plot(t, alpha, 'r')
        ax[2].plot(t, rews, 'g')
        ax[3].plot(t, actions, 'y')
        ax[0].set_title('Theta')
        ax[1].set_title('Alpha- pendulum angle')
        ax[2].set_title('Reward')
        ax[3].set_title('Actions')
    env.close()



if __name__ == '__main__':
    max_time_steps = 1e6
    num_cpu = 20

    env = SubprocVecEnv([make_env('env', i) for i in range(num_cpu)])

    eval_env = gym.make('Qube-100-v0')
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='./logs/a2c_swing/',
                                 log_path='./logs/a2c_swing/',
                                 eval_freq=int(2500),
                                 deterministic=True, render=False)

    model = A2C('MlpPolicy', env, verbose=1, device='cpu',
                tensorboard_log="./ppo_swing_sparse_tensorboard/", gamma=0.999)

    s = time.time()
    model.learn(total_timesteps=max_time_steps,
                tb_log_name="a2c_swing_rand_alpha", callback=eval_callback)

    e = time.time()-s
    print('Took {} seconds to train for {} steps'.format(e, max_time_steps))
    env_test = gym.make('Qube-100-v0')
    test_env(env_test, model)
    time.sleep(1)
    mean_reward, std_reward = evaluate_policy(model, env_test,
                                              n_eval_episodes=5)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
