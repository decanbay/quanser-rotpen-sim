import time
import numpy as np
import gym
import quanser_robots


class MetronomCtrl:
    """Rhythmically swinging metronome."""

    def __init__(self, u_max=2.0, f=0.5, dur=5.0):
        """
        Constructor

        :param u_max: maximum voltage
        :param f: frequency in Hz
        :param dur: task finishes after `dur` seconds

        """
        self.done = False
        self.u_max = u_max
        self.f = f
        self.dur = dur
        self.start_time = None

    def __call__(self, _):
        """
        Calculates the actions depending on the elapsed time.

        :return: scaled sinusoidal voltage
        """
        if self.start_time is None:
            self.start_time = time.time()
        t = time.time() - self.start_time
        if not self.done and t > self.dur:
            self.done = True
            u = 10. * np.sin(2 * np.pi * self.f * t)
        else:
            u = 10. * np.sin(2 * np.pi * self.f * t)
        return np.array([u])


def main():
    env = gym.make('DoublePendulumRR-v0')
    ctrl = MetronomCtrl()

    for i in range(2):
        print("\n\n###############################")
        print("Episode {0}".format(i))

        obs = env.reset()
        done = False

        print("\nStart Controller:\t\t\t", end="")
        while not done:
            env.render()
            act = 0.0 * ctrl(obs)
            obs, _, done, _ = env.step(np.array(act))

        print("Finished!")

    # Experiment Finished
    time.sleep(1.0)
    env.close()


if __name__ == "__main__":
    main()
