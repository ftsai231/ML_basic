from scipy import stats
import numpy as np


a = np.array([6, 8, 3, 0])


o = stats.mode(a)
print(o)
print(type(o))
o = o.mode[0]
print(o)

print(np.random.randint(0, 10, 3))