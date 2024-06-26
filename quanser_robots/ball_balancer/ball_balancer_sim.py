import numpy as np

from .base import BallBalancerBase, BallBalancerDynamics
from ..common import LabeledBox, VelocityFilter


class BallBalancerSim(BallBalancerBase):
    """
    Quanser 2 DoF Ball Balancer simulator class.
    """
    def __init__(self, fs, fs_ctrl, simplified_dyn=False):
        """
        :param simplified_dyn: flags if a dynamics model without Coriolis forces and without friction should be used
        """
        super().__init__(fs, fs_ctrl)
        self._dyn = BallBalancerDynamics(dt=self.timing.dt, simplified_dyn=simplified_dyn)
        init_state_min = np.array([0.7 * self._dyn.l_plate/2., 0,
                                   -0.05 * self.state_space.high[6], -0.05 * self.state_space.high[7]])
        init_state_max = np.array([0.8 * self._dyn.l_plate/2., 2*np.pi,
                                   0.05 * self.state_space.high[6], 0.05 * self.state_space.high[7]])
        self.init_space = LabeledBox(
            labels=('r', 'phi', 'vel_x', 'vel_y'),
            low=init_state_min, high=init_state_max, dtype=np.float32)

    def reset(self, init_state=None):
        """
        Reset the simulator
        :param init_state:
        :return: observation and false done flag
        """
        super().reset()  # sets the state to zero
        # Set a random initial state or a fixed one if specified
        if init_state is None:
            # init_space_sample = self.init_space.sample()  # uniformly
            init_space_sample = self._np_random.uniform(low=self.init_space.low,
                                                        high=self.init_space.high,
                                                        size=self.init_space.low.shape).astype(np.float32)
            # Initialize ball pos on plate on a circle
            self._state[2] = init_space_sample[0] * np.cos(init_space_sample[1])  # init ball x pos
            self._state[3] = init_space_sample[0] * np.sin(init_space_sample[1])  # init ball y pos
            self._state[6] = init_space_sample[2]  # init ball x vel
            self._state[7] = init_space_sample[3]  # init ball y vel
        else:
            if isinstance(init_state, np.ndarray):
                self._state = init_state.copy()
            else:
                try:
                    self._state = np.array(init_state)
                except Exception:
                    raise TypeError("Can not convert init_state to numpy array!")
        # print("Reset simulator to state {}".format(self._state))

        # Initialize velocity filter
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0],
                                        dt=self.timing.dt,
                                        x_init=self._state[:4].copy())

        # Return exact state (not estimated by velocity filter)
        return self._state.copy()

    def step(self, action):
        """
        Perform one simulation step
        :param action: agents action in the environment
        :return: tuple of observation, reward, done-flag, and environment info
        """
        assert action is not None, "Action should be not None"
        assert isinstance(action, np.ndarray), "The action should be a ndarray"
        assert not np.isnan(action).any(), "Action NaN is not a valid action"
        assert action.ndim == 1, "The action = {1} must be 1d but the input is {0:d}d".format(action.ndim, action)

        info = {'action_raw': action}
        # Add a bit of noise to action for robustness
        # action = action + 1e-6 * np.float32(self._np_random.randn(self.action_space.shape[0]))

        # Apply action limits
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._curr_action = action

        # Compute the dynamics
        accs, plate_angvel = self._dyn(self._state, self._plate_angs, action)

        # Integration step (symplectic Euler)
        self._state[4:] += accs * self.timing.dt  # next velocity
        self._state[:4] += self._state[4:] * self.timing.dt  # next position

        # Integration step (forward Euler, since we get the plate's angular velocities from the kinematics)
        self._plate_angs += plate_angvel * self.timing.dt  # next position

        # Also use the velocity filter in simulation (only the positions are measurable)
        pos = self._state[:4]
        vel = self._vel_filt(pos)
        obs = np.concatenate([pos, vel])

        reward = self._rew_fcn(obs, action)
        done = self._is_done()  # uses the simulation state and not the observation
        info.update({'state': self._state.copy()})

        self._step_count += 1
        return obs, reward, done, info

    def render(self, mode='human'):
        super().render(mode)

    def close(self):
        pass
