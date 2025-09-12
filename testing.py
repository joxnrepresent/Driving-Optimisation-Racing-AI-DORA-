import numpy as np
from pygame import Vector2
actions = [0.8, -99]
actions = np.array(actions)
actions = np.clip(actions, -1, 1)
print(actions)