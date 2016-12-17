import numpy as np
from time import time
def myfunc(a, b):
    "Return a-b if a>b, otherwise return a+b"
    if a > b:
        return a - b + c
    else:
        return a + b + c

t = time()
vfunc = np.vectorize(myfunc)
vfunc(np.arange(1E8), np.arange(1E8), 5)
print("Time for vectorize: {0} seconds".format(time()-t))

t = time()
for i in np.arange(1E8):
    myfunc(i,i,5)
print("Time for loops: {0} seconds".format(time()-t))
