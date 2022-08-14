# from rubics_cube_env import RubicsCubeEnv
from cross_solver.cross_env import CrossEnv
import time
import cv2
env = CrossEnv()
env.reset()
image = env.render()
cv2.imshow("Rubics cube uv", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
