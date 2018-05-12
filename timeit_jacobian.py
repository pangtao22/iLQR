import timeit
setup = '\
from quadrotor3D import CalcF, n, m, mass, g; \
from pydrake.forwarddiff import jacobian; \
import numpy as np; \
x0 = np.zeros(n); \
u0 = np.zeros(m); \
u0[:] = mass*g/4; \
x_u = np.hstack((x0, u0));'
N = 5000
t = timeit.timeit('jacobian(CalcF, x_u)', setup = setup, number = N)
t = timeit.timeit('QD.CalcPartials(x_u)', setup = setup, number = N)
print "average time:", t/N



#%%
import timeit
setup2 = '\
import julia; \
import numpy as np; \
julia.Julia().include(\'quadrotor_dynamics.jl\'); \
from julia import QuadrotorDynamics as QD; \
x0 = np.zeros(12); \
u0 = np.zeros(4); \
u0[:] = QD.mass*QD.g/4; \
x_u = np.hstack((x0, u0));'
N = 5000
t = timeit.timeit('QD.CalcPartials(x_u)', setup = setup2, number = N)
print "average time:", t/N
