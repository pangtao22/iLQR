import pydrake.examples.quadrotor as qd
import numpy as np
from quadrotor3D import *

#%% test if Eigen autodiff and pydrake.forwarddiff generates the same jacobian.
J2 = np.zeros((n,n+m))
x_u = np.zeros(n+m)
x_u[n:n+m] = 1.25

x_u = np.random.rand(n+m)

J1 = jacobian(CalcF, x_u)
qd.CalcJacobian(x_u, J2)

print np.linalg.norm(J1-J2)


