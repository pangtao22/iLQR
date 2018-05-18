import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from iLQR import DiscreteTimeIterativeLQR, WayPoint, TrajectorySpecs

#%% initilization
# discrete double integrator dynamics
#A = np.array([[1, h], [0, 1]]) 
#B = np.array([[h**2/2], [h]])

n = 2 # number of states
m = 1 # number of inputs

def CalcF(x_u):
  assert(x_u.size == m+n)
  x = x_u[0:n]
  u = x_u[n:n+m]    
  A = np.array([[0, 1], [0, 0]]) 
  B = np.array([[0], [1]])
  x_dot = A.dot(x) + B.dot(u)
  return x_dot

planner= DiscreteTimeIterativeLQR(CalcF, n, m)
#%% iLQR
h = 0.01 # time step.
N = 300 # horizon
x0 = np.array([0.,0,])
u0 = np.array([0.])

# desired fixed point
xd = np.array([1.,0])
ud = np.array([0.])

# cost weights
QN = 100*np.diag([1,0.1])
Q = np.diag([0.1, 0.1])# lqr cost
R = 0.1*np.eye(m) # lqr cost
W1 = 10*np.diag([1.0, 1.0])

# waypoints
x1 = np.array([0.3, 0.2])
t1 = h*N*0.4
rho1 = 100
xw = WayPoint(x1, t1, W1, rho1)

traj_specs = TrajectorySpecs(x0, u0, xd, ud, h, N, Q, R, QN, [xw])
traj_specs2 = TrajectorySpecs(x0, u0, xd, ud, h, N, Q, R, QN)
x, u, J, QN, Vx, Vxx, k, K = planner.CalcTrajectory(traj_specs)
x2, u2, J2, QN2, Vx2, Vxx2, k2, K2 = planner.CalcTrajectory(traj_specs2)
    
#%% plot trajectory with and without waypoint
i_x = -1 # which iteration to plot

t = np.array([i*h for i in range(N+1)])
fig = plt.figure(figsize=(4,6), dpi = 200)
ax_x = fig.add_subplot(311)
ax_x.set_ylabel("x")
ax_xdot = fig.add_subplot(312)
ax_xdot.set_ylabel("xdot")
ax_u = fig.add_subplot(313)
ax_u.set_ylabel("u")
ax_u.set_xlabel("t")
# reference lines at 0
ax_x.axhline(xd[0], color='r', ls='--')
ax_xdot.axhline(xd[1], color='r', ls='--')
ax_u.axhline(color='r', ls='--')

ax_x.plot(xw.t, xw.x[0], 'r*')
ax_xdot.plot(xw.t, xw.x[1], 'r*')


ax_x.plot(t, x[i_x,:,0], label = 'w/ way point')
ax_x.plot(t, x2[i_x,:,0], label = 'w/o way point')
ax_x.legend()

ax_xdot.plot(t, x[i_x,:,1], label = 'w/ way point')
ax_xdot.plot(t, x2[i_x,:,1], label = 'w/o way point')
ax_xdot.legend()

ax_u.plot(t[0:-1], u[i_x,:,0], label = 'w/ way point')
ax_u.plot(t[0:-1], u2[i_x,:,0], label = 'w/o way point')
plt.tight_layout()
ax_u.legend()

planner.PlotCosts(x[i_x], u[i_x], xd, ud, Q, R, QN, [xw], h)    


#%% plots of gradient and eignevalues of V
fig2 = plt.figure(figsize = (6,6), dpi = 200)
ax_phase = fig2.add_subplot(211)
ax_phase.set_xlabel('x')
ax_phase.set_ylabel('xdot')
ax_phase.set_aspect('equal')
ax_phase.plot(x[i_x,:,0], x[i_x,:,1])
ax_phase.quiver(x[i_x,:,0], x[i_x,:,1], -Vx[:,0], -Vx[:,1])

eigval_V = np.zeros((N+1, 2))
for i in range(N+1):
  eigvals = np.linalg.eigvals(Vxx[i])
  eigval_V[i,0] = max(eigvals)
  eigval_V[i,1] = min(eigvals)
ax_eigen = fig2.add_subplot(212)
ax_eigen.plot(t, eigval_V[:,0])
ax_eigen.plot(t, eigval_V[:,1])
ax_eigen.set_xlabel('time (s)')
ax_eigen.set_ylabel('eigenvalues of Vxx')
plt.tight_layout()
plt.show()

#%% 3D plot of different costs
i_x = -1
planner.traj_specs = traj_specs
J_traj = np.zeros(N+1)
J_lqr_traj = np.zeros(N+1)
J_wpt_traj = np.zeros(N+1)
discount_traj = np.zeros(N+1)
for i in range(N+1):
  J_lqr_traj[i] = planner.CalcLqrCost(x[i_x], u[i_x], i)
  J_wpt_traj[i] = planner.CalcWayPointsCost(x[i_x], i, 0)
  J_traj[i] = planner.CalcJ(x[i_x], u[i_x], 0, i)
  discount_traj[i] = planner.discount(planner.traj_specs.xw_list[0], i, 0)
      
fig = plt.figure(figsize = (6,6), dpi = 200)
ax_J = fig.add_subplot(111, projection = '3d')
ax_J.set_xlabel('x')
ax_J.set_ylabel('xdot')

l1, = ax_J.plot(x[i_x,:,0], x[i_x,:,1], J_traj, label = 'J_total')
l2, = ax_J.plot(x[i_x,:,0], x[i_x,:,1], J_lqr_traj, label = 'J_lqr')
l3, = ax_J.plot(x[i_x,:,0], x[i_x,:,1], J_wpt_traj, label = 'J_wpt')
l4, = ax_J.plot(x[i_x,:,0], x[i_x,:,1], '--', label = 'phase trajectory')

#scale = max(J_wpt_traj)/max(discount_traj)
#l5, = ax_J.plot(x[i_x,:,0], x[i_x,:,1], scale*discount_traj, label = 'discount_value')
#wpt = planner.traj_specs.xw_list[0]
#ax_J.plot([wpt.x[0], wpt.x[0]], [0,0.5], 'r--')

#idx_wpt = int(wpt.t/planner.traj_specs.h)
#ax_J.plot([x[i_x,idx_wpt,0]], [x[i_x,idx_wpt,1]], 'ro')
#idx = range(0, N+1,  5)
ax_J.plot([xw.x[0]], [xw.x[1]], [0], 'ro')

plt.legend()

plt.tight_layout()
plt.show()





































