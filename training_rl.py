# -*- coding: utf-8 -*-

import numpy as np
import gym
import time
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import argparse
from quanser_robots import GentlyTerminating
from stable_baselines3.common import results_plotter

alg_dict = {'PPO':PPO, 'A2C':A2C, 'DQN':DQN}

# env = RotpenSwingupSparseEnv()

from typing import Callable

def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = GentlyTerminating(gym.make('Rotpen-100-v0'))
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Pass your arguments')
    parser.add_argument('--alg', type = str, default = 'PPO')
    parser.add_argument('--num_cpu', type = int, default =24)
    args = parser.parse_args()
    print(args)
    num_cpu = args.num_cpu  # Number of processes to use
    # Create the vectorized environment
    alg = args.alg
    if alg=='DQN':
        num_cpu=1
    if num_cpu ==1:
        print('Num_CPU = 1')
        envs = GentlyTerminating(gym.make('Rotpen-100-v0'))
    else:
        envs = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # model = A2C('MlpPolicy', env, verbose=0)
    alg2 = alg_dict[alg]

    print(alg2)
    model = alg2('MlpPolicy', envs, verbose=1,device='cpu')
    env_id = 'Rotpen-100-v0'
    eval_env = gym.make(env_id)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
    
    s_time = time.time()
    total_timesteps=num_cpu*2048*500
    model.learn(total_timesteps=total_timesteps)
    t_time = time.time()-s_time
    print('Totoal time passed = {} seconds'.format(t_time))
    print('{} it/sec'.format(total_timesteps/t_time))
    
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
    time.sleep(1)
    fname = 'rotpen_'+ alg
    model.save(fname)

    results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, env_id)
    
    time.sleep(1)
    test_env = GentlyTerminating(gym.make('Rotpen-500-v0'))
    obs = test_env.reset()
    for i in range(2500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        if i%250==0:
            print(i)
        if done:
            print('Done')
            obs = test_env.reset()
    test_env.close()