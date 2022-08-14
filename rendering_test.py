from rubics_cube_env import RubicsCubeEnv
import time
import cv2
env = RubicsCubeEnv()
env.reset()
# R R L U` F B` D L` U U R` F B` D
actions = [6, 6, 2, 1, 8, 5, 10, 3, 0, 0, 7, 8, 5, 10]
for a in actions:
    state, _, __, ___, = env.step(a)

image = env.render()
cv2.imshow("Rubics cube uv", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
