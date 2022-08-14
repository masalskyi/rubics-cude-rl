from rubics_cube_env import RubicsCubeEnv


class CrossEnv(RubicsCubeEnv):
    def reset(self, *, seed=None, return_info: bool = False, options=None):
        obs = super(CrossEnv, self).reset()
        for i in range(21):
            obs = self.perform_action(obs, self.action_space.sample())
        self.state = obs
        self.length = 10
        return self.state

    def step(self, action):
        obs, reward, done, info = super(CrossEnv, self).step(action)
        reward = -1
        self.length -= 1
        if obs[1] == 1:
            reward += 10
            done = True
        if self.length == 0:
            done = True
        return obs, reward, done, info
