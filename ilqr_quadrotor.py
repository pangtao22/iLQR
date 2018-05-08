import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from iLQR import DiscreteTimeIterativeLQR, WayPoint, TrajectorySpecs
#%% initilization
n = 6 # number of states. q = [x,y,theta], x = [q, q_dot]
m = 2 # number of inputs

# dynamics
def CalcF(x_u):
    # m, l, g, I = 1
    assert(x_u.size == m+n)
    x = x_u[0:n]
    u = x_u[n:n+m]    
    theta = x[2]
    
    x_dotdot = x
    x_dotdot[0] = x[3]
    x_dotdot[1] = x[4]
    x_dotdot[2] = x[5]
    x_dotdot[3] = -sin(theta) * (u[0] + u[1])
    x_dotdot[4] = cos(theta) * (u[0] + u[1]) - 1 # 1 = mg
    x_dotdot[5] = u[1] - u[0] # l = 1

    return x_dotdot

planner= DiscreteTimeIterativeLQR(CalcF, n, m)
#%% iLQR
h = 0.01 # time step.
N = 600 # horizon
x0 = np.array([0,0,0,0,0,0])
u0 = np.array([0.5, 0.5])

# desired fixed point
xd = np.array([1,0,0,0,0,0])
ud = np.array([0.5, 0.5])

# cost weights
QN = np.diag([100,10,10,0.1,0.1,10])
Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])# lqr cost
R = np.eye(m) # lqr cost
W1 = 1*np.diag([1., 1, 0., 0., 0., 0.])

# waypoints
x1 = np.array([0.6, 0., 0, 0, 0, 0])
t1 = h*N*0.5
rho1 = 5
xw = WayPoint(x1, t1, W1, rho1)

traj_specs = TrajectorySpecs(x0, u0, xd, ud, h, N, Q, R, QN)

Ni = 2
x, u, J, QN, Vx, Vxx, k, K =\
    planner.CalcTrajectory(traj_specs, Ni)

    
#%% plot
t = np.array([i*h for i in range(N+1)])
fig = plt.figure(figsize=(6,16), dpi = 100)
ax_x = fig.add_subplot(411)
ax_x.set_ylabel("x")
ax_y = fig.add_subplot(412)
ax_y.set_ylabel("y")
ax_theta = fig.add_subplot(413)
ax_theta.set_ylabel("theta")
ax_u = fig.add_subplot(414)
ax_u.set_ylabel("u")
ax_u.set_xlabel("t")
# reference lines at 0
ax_x.axhline(color='r', ls='--')
ax_y.axhline(color='r', ls='--')
ax_theta.axhline(color='r', ls='--')
ax_u.axhline(color='r', ls='--')

for i in range(Ni+1):
    ax_x.plot(t, x[i,:,0])
    ax_x.plot(xw.t, xw.x[0], 'r*')
    ax_y.plot(t, x[i,:,1])
    ax_y.plot(xw.t, xw.x[1], 'r*')
    ax_theta.plot(t, x[i,:,2])
    ax_theta.plot(xw.t, xw.x[2], 'r*')
    ax_u.plot(t[0:-1], u[i,:,0])
plt.show()

planner.PlotCosts(x[-1], u[-1], xd, ud, Q, R, QN, [xw], h)
      
    
#%% simulate and plot
# broken
#dt = 0.001
#T = 20000
#t = dt*np.arange(T+1)
#x = np.zeros((T+1, n))
#x[0] = [0, 0, 0.1, 0, 0, 0]
#
#for i in range(T):
#    x_u = np.hstack((x[i], -K0.dot(x[i]-xd) + ud))
#    x[i+1] = x[i] + dt*CalcF(x_u)
#    
#fig = plt.figure(figsize=(6,12), dpi = 100)
#
#ax_x = fig.add_subplot(311)
#ax_x.set_ylabel("x")
#ax_x.plot(t, x[:,0])
#ax_x.axhline(color='r', ls='--')
#
#ax_y = fig.add_subplot(312)
#ax_y.set_ylabel("y")
#ax_y.plot(t, x[:,1])
#ax_y.axhline(color='r', ls='--')
#
#ax_phase = fig.add_subplot(313)
#ax_phase.set_ylabel("theta")
#ax_phase.set_xlabel("t")
#ax_phase.plot(t, x[:,2])
#ax_phase.axhline(color='r', ls='--')




























