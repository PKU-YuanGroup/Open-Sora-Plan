import random 
import numpy as np


randnum = np.random.randint(0, 1e8, size=200)

with open('seed.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in randnum]))