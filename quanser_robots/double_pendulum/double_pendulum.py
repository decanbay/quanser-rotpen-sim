import numpy as np
from quanser_robots.common import VelocityFilter, PhysicSystem, Simulation, Timing, NoFilter
from quanser_robots.double_pendulum.base import DoublePendulumBase


class DoublePendulum(Simulation, DoublePendulumBase):

    def __init__(self, fs, fs_ctrl, long_pole=False):

        DoublePendulumBase.__init__(self, fs, fs_ctrl)

        Simulation.__init__(self, fs,
                            fs_ctrl,
                            dynamics=DoublePendulumDynamics(long=long_pole),
                            entities=['x', 'theta1', 'theta2'],
                                        filters={
                                            'x': lambda x_init: NoFilter(x_init=x_init, dt=self.timing.dt),
                                            'theta1': lambda x_init: NoFilter(x_init=x_init, dt=self.timing.dt),
                                            'theta2': lambda x_init: NoFilter(x_init=x_init, dt=self.timing.dt)
                                        },

                                      initial_distr={
                                          'x': lambda: 0.,
                                          'theta1': lambda: 0.05 * np.random.uniform(-1.,1.),
                                          'theta2': lambda: 0.05 * np.random.uniform(-1.,1.)
                                      })

        # Transformations for the visualization:
        self.cart_trans = None
        self.pole_trans = None
        self.pole2_trans = None
        self.track = None
        self.axle = None

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = 2. * self._dynamics._x_lim
        scale = screen_width / world_width

        pole_width = scale * 0.04 * self._dynamics.l1
        pole_len1 = scale * self._dynamics.l1
        pole_len2 = scale * self._dynamics.l2

        cart_y = 100                                        # TOP OF CART
        cart_width = scale * 0.3 * self._dynamics.l1
        cart_height = scale * 0.2 * self._dynamics.l1

        if self.viewer is None:

            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Visualize Cart:
            l, r, t, b = -cart_width/2, cart_width/2, cart_height/2, -cart_height/2
            axle_offset = cart_height/4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # Visualize Pendulum
            l, r, t, b = -pole_width/2, pole_width/2, pole_len1-pole_width/2, -pole_width/2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8,.6,.4)
            self.pole_trans = rendering.Transform(translation=(0, axle_offset))

            l, r, t, b = -pole_width/2, pole_width/2, pole_len2-pole_width/2, -pole_width/2
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(.8, .6, .4)
            self.pole2_trans = rendering.Transform(translation=(0, axle_offset))

            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            pole2.add_attr(self.pole2_trans)
            pole2.add_attr(self.cart_trans)

            self.viewer.add_geom(pole)
            self.viewer.add_geom(pole2)

            # Visualize axle:
            self.axle = rendering.make_circle(pole_width/2)
            self.axle.add_attr(self.pole_trans)
            self.axle.add_attr(self.cart_trans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)

            # Visualize track:
            self.track = rendering.Line((0, cart_y), (screen_width, cart_y))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self._sim_state is None:
            return None

        # Update the visualization:
        x = self._sim_state
        cart_x = x[0] * scale + screen_width / 2.0
        self.cart_trans.set_translation(cart_x, cart_y)
        self.pole_trans.set_rotation(x[1])
        self.pole2_trans.set_rotation(x[2] + x[1])
        self.pole2_trans.set_translation(-pole_len1 * np.sin(x[1]), pole_len1 * np.cos(x[1]))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class DoublePendulumDynamics:

    def __init__(self, long=False):
        self._x_lim = 0.814 / 2.        # Limit of position of the cart [m]
        self._g = 9.81                  # Gravitational acceleration [m/s^2]
        self._eta_m = 1.                # Motor efficiency  []
        self._eta_g = 1.                # Planetary Gearbox Efficiency []
        self._Kg = 3.71                 # Planetary Gearbox Gear Ratio
        self._Jm = 3.9E-7               # Rotor inertia [kg.m^2]
        self._r_mp = 1.30E-3            # Motor Pinion radius [m] #TODO: was 6.35E-3
        self._Rm = 2.6                  # Motor armature Resistance [Ohm]
        self._Kt = .00767               # Motor Torque Constant [N.zz/A]
        self._Km = .00767               # Motor Torque Constant [N.zz/A]
        self._mc = 0.38                 # Mass of the cart [kg]
        self._Beq = 4.3                 # Equivalent Viscous damping Coefficient

        self.m1 = 0.072                 # Mass of pole 1 [kg]
        self.m2 = 0.127                 # Mass of pole 2 [kg]
        self.l1 = 0.1143                # Length Pole 1 [m]
        self.l2 = 0.1778                # Half Length Pole 2 [m]
        self.Bp1 = self.Bp2 = 0.0024    # Viscous coefficient at the pole

    def __call__(self, s, V_m):
        x, alpha1, alpha2, x_dot, alpha1_dot, alpha2_dot = s

        # Transformation to the system used in the dynamics
        theta1 = -alpha1
        theta2 = -alpha2-alpha1
        theta1_dot = -alpha1_dot
        theta2_dot = -alpha2_dot - alpha1_dot

        # Compute force acting on the cart:
        F = (self._eta_g * self._Kg * self._eta_m * self._Kt) / (self._Rm * self._r_mp) * \
            (-self._Kg * self._Km * x_dot / self._r_mp + self._eta_m * V_m)

        m, m1, m2 = self._mc, self.m1, self.m2
        l1, l2 = self.l1, self.l2
        Bp1, Bp2 = self.Bp1, self.Bp2
        Beq = self._Beq

        A = np.array([[m+m1+m2, l1 * (m1+m2) * np.cos(theta1), m2 * l2 * np.cos(theta2)],
                      [l1 * (m1+m2) * np.cos(theta1), l1**2 * (m1+m2), l1 * l2 * m2 * np.cos(theta1-theta2)],
                      [l2 * m2 * np.cos(theta2), l1*l2*m2*np.cos(theta1-theta2), l2**2*m2]
                      ], dtype=np.float64)

        b = np.array([+l1 * (m1+m2) * theta1_dot**2 * np.sin(theta1) + m2 * l2 * theta2_dot**2 * np.sin(theta2) + F - Beq * x_dot,
                      -l1 * l2 * m2 * theta2_dot**2 * np.sin(theta1-theta2) + self._g * (m1+m2) * l1 *np.sin(theta1) - Bp1 * theta1_dot,
                      +l1 * l2 * m2 * theta1_dot**2 * np.sin(theta1-theta2) + self._g * l2 * m2 * np.sin(theta2) - Bp2 * theta2_dot
                      ], dtype=np.float64)

        x_ddot, theta1_ddot, theta2_ddot = np.linalg.solve(A, b)

        # Transformation to the system used externally
        alpha1_ddot = -theta1_ddot
        alpha2_ddot = -theta2_ddot - alpha1_ddot

        return np.array([x_ddot, alpha1_ddot, alpha2_ddot])

