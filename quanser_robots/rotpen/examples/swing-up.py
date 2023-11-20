"""
The minimal program that shows the basic control loop on the simulated swing-up.
"""

import gym
from quanser_robots import GentlyTerminating
from quanser_robots.rotpen import SwingUpCtrl

env = GentlyTerminating(gym.make('Rotpen-250-v0'))

ctrl = SwingUpCtrl()
obs = env.reset()
done = False
while not done:
    env.render()
    act = ctrl(obs)
    obs, _, done, _ = env.step(act)

env.close()
