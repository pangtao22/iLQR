import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from iLQR import IterativeLQR


#%% initilization
# discrete double integrator dynamics
#A = np.array([[1, h], [0, 1]]) 
#B = np.array([[h**2/2], [h]])

n = 2 # number of states
m = 1 # number of inputs

A = np.array([[0, 1], [0, 0]]) 
B = np.array([[0], [1]])
def CalcF(x_u):
    assert(x_u.size == m+n)
    x = x_u[0:n]
    u = x_u[n:n+m]    
    x_dotdot = A.dot(x) + B.dot(u)

    return x_dotdot

planner= IterativeLQR(CalcF, n, m)
#%% iLQR
h = 0.01 # time step.
N = 500 # horizon
x0 = np.array([1,0])
Q = 1000*np.diag([1, 1]) # lqr cost
R = np.eye(1) # lqr cost
xd = np.zeros(n)
ud = np.zeros(m)
Ni = 1
x_new, u_new, x, u, J, QN =\
    planner.CalcTrajectory(x0 , xd, ud, h, N, Q, R, Ni)
    
    
#%% plot
t = np.array([i*h for i in range(N+1)])
fig = plt.figure(figsize=(6,12), dpi = 100)

ax_x = fig.add_subplot(311)
ax_x.set_ylabel("x")
ax_x.plot(t, x_new[:,0])
ax_x.plot(t, x[:,0])
ax_x.axhline(color='r', ls='--')

ax_y = fig.add_subplot(312)
ax_y.set_ylabel("xdot")
ax_y.plot(t, x_new[:,1])
ax_y.plot(t, x[:,1])
ax_y.axhline(color='r', ls='--')

ax_u = fig.add_subplot(313)
ax_u.set_ylabel("u")
ax_u.set_xlabel("t")
ax_u.plot(t[0:-1], u_new)
ax_u.plot(t[0:-1], u)
ax_u.axhline(color='r', ls='--')

plt.show()




























