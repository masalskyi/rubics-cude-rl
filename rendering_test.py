from rubics_cube_env import RubicsCubeEnv
import time
import cv2
env = RubicsCubeEnv()
env.reset()
# F R U B L`
actions = [8, 6, 0, 4, 3]
for a in actions:
    state, _, __, ___, = env.step(a)

image = env.render()
cv2.imshow("Rubics cube uv", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
