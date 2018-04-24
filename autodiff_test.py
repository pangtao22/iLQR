from pydrake.forwarddiff import jacobian
import numpy as np
from numpy import sin,cos

def f(x):
    return np.array([x[0] + x[1]])


x = np.array([0,1])
fx = jacobian(f,x)

print fx
