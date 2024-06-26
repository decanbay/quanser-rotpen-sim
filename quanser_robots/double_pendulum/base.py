import numpy as np
import warnings
import gym
from gym.utils import seeding

np.set_printoptions(precision=6, suppress=True)

import numpy as np
from quanser_robots.common import LabeledBox, Base, Timing
np.set_printoptions(precision=6, suppress=True)


class DoublePendulumBase(Base):

    def __init__(self, fs, fs_ctrl):

        super(DoublePendulumBase, self).__init__(fs, fs_ctrl)
        self._state = None

        # Constants
        self.th_lim = 0.25                  # [rad]
        self._length = 0.814                # [m]
        self._x_lim = self._length / 2.     # [m]
        act_max = np.array([24.0])          # [V]

        sens_max = np.array([np.inf, np.pi, np.pi])
        state_max = np.array([self._x_lim, np.pi, self.th_lim, np.inf, np.inf, np.inf])
        obs_max = np.array([self._x_lim , np.pi, self.th_lim, np.inf, np.inf, np.inf])

        # Spaces
        self.sensor_space = LabeledBox(
            labels=('x', 'theta1', 'theta2'),
            low=-sens_max, high=sens_max, dtype=np.float32)

        self.state_space = LabeledBox(
            labels=('x', 'theta1', 'theta2', 'x_dot', 'theta1_dot', 'theta2_dot'),
            low=-state_max, high=state_max, dtype=np.float32)

        self.observation_space = LabeledBox(
            labels=('x', 'theta1', 'theta2', 'x_dot', 'theta1_dot', 'theta2_dot'),
            low=-obs_max, high=obs_max, dtype=np.float32)

        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float32)

        self.reward_range = (0., 2*self.th_lim**2)

        # Function to ensure that state and action constraints are satisfied:
        self._lim_act = ActionLimiter()

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _limit_act(self, action):
        if np.abs(action) > 24.:
            warnings.warn("Control signal a = {0:.2f}V should be between -24V and 24V.".format(action))

        return np.clip(action, -24., 24.)

    def _zero_sim_step(self):
        return self._sim_step(np.array([0.0]))

    def _rwd(self, x, a):

        x_c, th1, th2, _, _, _ = x

        rwd = -(th1**2 + th2**2)

        done = np.abs(th1) > self.th_lim or np.abs(th2) > self.th_lim or np.abs(x_c) > self._x_lim
        return np.float32(rwd) + 2 * self.th_lim**2, done

    def _observation(self, state):
        """
        A observation is provided given the internal state.

        :param state: (x, theta_0, theta_1, x_dot, theta0_dot, theta1_dot)
        :type state: np.array
        :return: (x, theta_0, theta_1, x_dot, theta0_dot, theta1_dot)
        :rtype: np.array
        """
        return state


class ActionLimiter:

    def __init__(self):
        pass

    def _joint_lim_violation_force(self, x):
        pass

    def __call__(self, x, a):
        return a






