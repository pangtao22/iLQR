import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from iLQR import DiscreteTimeIterativeLQR, WayPoint, TrajectorySpecs
from quadrotor3D import (CalcF, PlotTrajectoryMeshcat, n, m, mass, g, PlotTraj, 
                         PlotTrajTimeLapse)
import meshcat
#%% initilization
planner= DiscreteTimeIterativeLQR(CalcF, n, m)
#%% iLQR
h = 0.01 # time step.
N = 200 # horizon
x0 = np.zeros(n)
u0 = np.zeros(m)
u0[:] = mass * g / 4

# desired fixed point
xd = np.zeros(n)
xd[0:3] = [3.,-1, 1]
ud = np.zeros(m)
ud[:] = u0

# costs
QN = 100*np.diag([10,10,10,1,1,1,  0.1,0.1,0.1,0.1,0.1,0.1])
Q_vec = np.ones(n)
Q_vec[6:12] *= 0.1
Q = np.diag(Q_vec)# lqr cost
R = np.eye(m) # lqr cost

# waypoints
x1 = np.zeros(n)
x1[0:3] = [1, -0.3, -0.3]
x1[3] = np.pi/6;
t1 = 0.8
W1 = np.zeros(n)
W1_vec = np.zeros(n)
W1_vec[0] = 0.1
W1_vec[1] = 0.1
W1_vec[2] = 0.1
W1_vec[3] = 10
W1 = np.diag(W1_vec)
rho1 = 5
xw = WayPoint(x1, t1, W1, rho1)

traj_specs = TrajectorySpecs(x0, u0, xd, ud, h, N, Q, R, QN, xw_list=[xw])

if __name__ == "__main__":
    x, u, J, QN, Vx, Vxx, k, K = planner.CalcTrajectory(traj_specs)
    
    PlotTraj(x, h, [xw])
    
    wpts_list = np.zeros((len(traj_specs.xw_list)+1, 3))
    for i, xw in enumerate(traj_specs.xw_list):
        wpts_list[i] = xw.x[0:3]
    wpts_list[-1] = traj_specs.xd[0:3]
    #%% open meshcat
    vis = meshcat.Visualizer()
    vis.open

    #%% Meshcat animation
    t = [i*h for i in range(N+1)]
    PlotTrajectoryMeshcat(x[-1], t, vis, wpts_list)
    
    #%%
    PlotTrajTimeLapse(x[-1], 7, vis, wpts_list)