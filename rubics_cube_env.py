import gym
import numpy as np
import cv2


class RubicsCubeEnv(gym.Env):
    class StateTransition:
        # side :
        # 0 - up (white),
        # 1 - left (orange),
        # 2 - back (blue),
        # 3 - right (red),
        # 4 - front (green),
        # 5 - button (yellow)
        def __init__(self, side):
            self.forward = {}
            self.backward = {}
            self.fill_transitions(side)

        def _fill_cycle(self, cycle):
            for i in range(1, len(cycle) + 1):
                self.forward[cycle[i - 1]] = cycle[i % len(cycle)]

        def _fill_circle(self, start):
            self._fill_cycle([start, start + 2, start + 4, start + 6])

        def fill_transitions(self, side):
            if side == 0:
                self._fill_circle(0)
                self._fill_circle(1)

                self._fill_cycle([8, 22, 30, 32])
                self._fill_cycle([9, 23, 31, 33])
                self._fill_cycle([10, 16, 24, 34])

            elif side == 1:
                self._fill_circle(8)
                self._fill_circle(9)

                self._fill_cycle([0, 32, 40, 18])
                self._fill_cycle([7, 39, 47, 17])
                self._fill_cycle([6, 38, 46, 16])

            elif side == 2:
                self._fill_circle(16)
                self._fill_circle(17)

                self._fill_cycle([0, 14, 44, 24])
                self._fill_cycle([1, 15, 45, 25])
                self._fill_cycle([2, 8, 46, 26])

            elif side == 3:
                self._fill_circle(24)
                self._fill_circle(25)

                self._fill_cycle([2, 20, 42, 34])
                self._fill_cycle([3, 21, 43, 35])
                self._fill_cycle([4, 22, 44, 36])

            elif side == 4:
                self._fill_circle(32)
                self._fill_circle(33)

                self._fill_cycle([6, 30, 42, 12])
                self._fill_cycle([5, 29, 41, 11])
                self._fill_cycle([4, 28, 40, 10])
            elif side == 5:
                self._fill_circle(40)
                self._fill_circle(41)

                self._fill_cycle([38, 28, 20, 14])
                self._fill_cycle([37, 27, 19, 13])
                self._fill_cycle([36, 26, 18, 12])
            for key in self.forward:
                self.backward[self.forward[key]] = key

    def __init__(self):
        super(RubicsCubeEnv, self).__init__()
        # 2*i - turn i side clockwise, 2*i+1 -turn i side counter_clock_wise
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(low=0, high=46, shape=(20,), dtype=np.int)
        self.state = np.array([0, 1, 2, 3, 4, 5, 6, 7, 38, 17, 21, 34, 39, 40, 41, 42, 43, 44, 45, 46])
        self.action_transitions = [self.StateTransition(side=i) for i in range(6)]

    def step(self, action):
        pass

    def reset(self, *, seed=None, return_info: bool = False, options=None):
        pass

    def render(self, mode="human"):
        pass
