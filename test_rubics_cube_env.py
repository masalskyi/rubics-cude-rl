import unittest
from rubics_cube_env import RubicsCubeEnv
import numpy as np


class RubicsBasicEnvTest(unittest.TestCase):
    def test_case_1(self):
        env = RubicsCubeEnv()
        state = env.reset()
        # F R U B L`
        actions = [8, 6, 0, 4, 3]
        for a in actions:
            state, _, __, ___, = env.step(a)
        self.assertTrue(
            (state == np.array([14, 3, 22, 23, 0, 33, 34, 13, 15, 19, 43, 37, 40, 9, 10, 35, 36, 25, 26, 39])).all())

    def test_case_2(self):
        env = RubicsCubeEnv()
        state = env.reset()
        # R R L U` F B` D L` U U R` F B` D
        actions = [6, 6, 2, 1, 8, 5, 10, 3, 0, 0, 7, 8, 5, 10]
        for a in actions:
            state, _, __, ___, = env.step(a)
        self.assertTrue(
            (state == np.array([14, 23, 38, 15, 26, 7, 28, 21, 11, 19, 47, 5, 8, 43, 4, 31, 10, 37, 24, 35])).all())


if __name__ == '__main__':
    unittest.main()
