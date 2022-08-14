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
            elif side == 6:
                self._fill_cycle([0, 8, 16])
                self._fill_cycle([2, 22, 24])
                self._fill_cycle([4, 30, 34])
                self._fill_cycle([6, 32, 10])
                self._fill_cycle([40, 12, 38])
                self._fill_cycle([42, 36, 28])
                self._fill_cycle([44, 26, 20])
                self._fill_cycle([46, 18, 14])

                self._fill_cycle([1, 23])
                self._fill_cycle([3, 31])
                self._fill_cycle([5, 33])
                self._fill_cycle([7, 9])
                self._fill_cycle([39, 11])
                self._fill_cycle([17, 15])
                self._fill_cycle([21, 25])
                self._fill_cycle([35, 29])
                self._fill_cycle([37, 41])
                self._fill_cycle([47, 13])
                self._fill_cycle([45, 19])
                self._fill_cycle([43, 27])

            for key in self.forward:
                self.backward[self.forward[key]] = key

    def __init__(self):
        super(RubicsCubeEnv, self).__init__()
        # 2*i - turn i side clockwise, 2*i+1 -turn i side counter_clock_wise
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(low=0, high=46, shape=(20,), dtype=int)
        self.state = np.array([0, 1, 2, 3, 4, 5, 6, 7, 39, 17, 21, 35, 40, 41, 42, 43, 44, 45, 46, 47])
        self.indices_of_detectors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 39, 17, 21, 35, 40, 41, 42, 43, 44, 45, 46, 47])
        self.action_transitions = [self.StateTransition(side=i) for i in range(6)]

        self.square_size = 25
        self.visual_indices = [
            [3, 3],
            [3, 4],
            [3, 5],
            [4, 5],
            [5, 5],
            [5, 4],
            [5, 3],
            [4, 3],

            [3, 2],
            [4, 2],
            [5, 2],
            [5, 1],
            [5, 0],
            [4, 0],
            [3, 0],
            [3, 1],

            [2, 3],
            [1, 3],
            [0, 3],
            [0, 4],
            [0, 5],
            [1, 5],
            [2, 5],
            [2, 4],

            [3, 6],
            [3, 7],
            [3, 8],
            [4, 8],
            [5, 8],
            [5, 7],
            [5, 6],
            [4, 6],

            [6, 3],
            [6, 4],
            [6, 5],
            [7, 5],
            [8, 5],
            [8, 4],
            [8, 3],
            [7, 3],

            [9, 3],
            [9, 4],
            [9, 5],
            [10, 5],
            [11, 5],
            [11, 4],
            [11, 3],
            [10, 3]
        ]
        self.colors = np.array([[255, 255, 255],
                                [255, 128, 0],
                                [0, 127, 255],
                                [255, 51, 51],
                                [0, 153, 77],
                                [255, 255, 102]], dtype=np.float32)
        self.center_indices = [[4, 4],
                               [4, 1],
                               [1, 4],
                               [4, 7],
                               [7, 4],
                               [10, 4]]
        self.element_transitions = self.StateTransition(6)
        self.element_colors = [
            [0, 1, 2],
            [0, 2],
            [0, 2, 3],
            [0, 3],
            [0, 3, 4],
            [0, 4],
            [0, 4, 1],
            [0, 1],
            [4, 1],
            [2, 1],
            [2, 3],
            [4, 3],
            [5, 1, 4],
            [5, 4],
            [5, 4, 3],
            [5, 3],
            [5, 3, 2],
            [5, 2],
            [5, 2, 1],
            [5, 1]
        ]

    def step(self, action):
        is_backward = action % 2 == 1
        action = action // 2
        new_state = np.copy(self.state)
        for i, v in enumerate(self.state):
            if v in self.action_transitions[action].forward:
                if is_backward:
                    new_state[i] = self.action_transitions[action].backward[v]
                else:
                    new_state[i] = self.action_transitions[action].forward[v]
        self.state = new_state
        return self.state, 0, False, {}

    def reset(self, *, seed=None, return_info: bool = False, options=None):
        self.state = np.array([0, 1, 2, 3, 4, 5, 6, 7, 39, 17, 21, 35, 40, 41, 42, 43, 44, 45, 46, 47])
        return self.state

    def render(self, mode="human"):
        image = np.ones((12 * self.square_size, 9 * self.square_size, 3), dtype=np.float32) * 255.0
        self._draw_centers(image)
        self._draw_squares(image)
        self._draw_lines(image)
        return image / 255.0
        # img = self._draw_edges(img)

    def _fill_square(self, image, y, x, color):
        image[y * self.square_size: (y + 1) * self.square_size,
        x * self.square_size: (x + 1) * self.square_size] = color

    def _draw_centers(self, image):
        for i, center in enumerate(self.center_indices):
            self._fill_square(image, center[0], center[1], self.colors[i])

    def _draw_squares(self, image):
        for i, colors in enumerate(self.element_colors):
            element = self.state[i]
            for j in range(len(colors)):
                y, x = self.visual_indices[element]
                color_indx = colors[j]
                self._fill_square(image, y, x, self.colors[color_indx])
                element = self.element_transitions.forward[element]

    def _draw_lines(self, image):
        for i in range(3, 7):
            image[:, i * self.square_size] = [0, 0, 0]

        for i in range(3, 7):
            image[i * self.square_size, :] = [0, 0, 0]

        for i in range(10):
            image[3 * self.square_size:6 * self.square_size, min(9 * self.square_size - 1, i * self.square_size)] \
                = [0, 0, 0]

        for i in range(13):
            image[min(12 * self.square_size - 1, i * self.square_size), 3 * self.square_size:6 * self.square_size] = \
                [0, 0, 0]

        image[:, 3 * self.square_size - 1] = [0, 0, 0]
        image[:, 3 * self.square_size + 1] = [0, 0, 0]

        image[:, 6 * self.square_size - 1] = [0, 0, 0]
        image[:, 6 * self.square_size + 1] = [0, 0, 0]

        image[3 * self.square_size - 1, :] = [0, 0, 0]
        image[3 * self.square_size + 1, :] = [0, 0, 0]

        image[6 * self.square_size - 1, :] = [0, 0, 0]
        image[6 * self.square_size + 1, :] = [0, 0, 0]

        image[9 * self.square_size - 1, 3 * self.square_size:6 * self.square_size] = [0, 0, 0]
        image[9 * self.square_size + 1, 3 * self.square_size:6 * self.square_size] = [0, 0, 0]
