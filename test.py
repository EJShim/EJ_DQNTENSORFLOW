
import numpy as np


weights = np.array([0, 3, 2, 5, 12, 5, 6, 7, 8, 9, 10, 11, 12, 13])

total_vars = len(weights)

for idx,var in enumerate(weights[0:total_vars//2]):
    print(idx, var)
