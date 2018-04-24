import numpy as np
from numpy import linalg as LA
from numpy import sin, cos
import matplotlib.pyplot as plt
# Notations in this code follow "Synthesis and stabilization of complex 
# behaviors through online trajectory optimization" by Y. Tassa and E. Todorov.

#%% dynamics and derivatives
# dynamics
def CalcF(x, u=np.zeros(1)):
    assert(x.size == 4)
    assert(u.size == 1)

    theta = x[1]
    xc_dot = x[2]
    theta_dot = x[3]
    
    x_dot = np.zeros(4)
    x_dot[0] = xc_dot
    x_dot[1] = theta_dot
    
    x_dot[2] = u[0] * theta_dot**2*sin(theta) + sin(theta)*cos(theta)
    x_dot[2] /= 1+sin(theta)**2
    
    x_dot[3] = -cos(theta)*u[0] - -theta_dot**2*sin(2*theta)/2 - 2*sin(theta)
    x_dot[3] /= 1+sin(theta)**2
    
    # damping
    # x_dot[2] += -0.1*xc_dot
    # x_dot[3] += -0.1*theta_dot
    
    return x_dot

# x derivatives
def CalcFx(x, u=np.zeros(1)):
    fx = np.zeros((4,4))
    fx[0:2][:,2:4] = np.eye(2)
 
    theta = x[1]
    thetad = x[3]
    
    s = sin(theta)
    c = cos(theta)
    s2 = sin(2*theta)
    c2 = cos(2*theta)
    a = 1+sin(theta)**2
    
    xdd_thetad = 2 * thetad * s / a
    thetadd_thetad = -thetad * s2 / a
    
    xdd_theta = (thetad**2*c + c2) * a - (u[0] + thetad**2*s + s2/2)*s2
    xdd_theta /= a**2
    
    thetadd_theta = (s*u[0] - thetad**2*c2 - 2*c)*a - \
                    (-c*u[0] - thetad**2*s2/2 - 2*s)*s2
    thetadd_theta /= a**2
    
    fx[2,1] = xdd_theta
    fx[2,3] = xdd_thetad 
    fx[3,1] = thetadd_theta
    fx[3,3] = thetadd_thetad
    
    return fx


# u derivatives
def CalcFu(x, u=np.zeros(1)):
    theta = x[1]   
    fu = np.array([0,0,1,-cos(theta)]) / (1+sin(theta)**2)
    
    return fu
    

#%% simulate and plot
#dt = 0.001
#T = 20000
#t = dt*np.arange(T+1)
#x = np.zeros((T+1, 4))
#x[0] = [0, np.pi/2, 0, 0]
#
#for i in range(T):
#    x[i+1] = x[i] + dt*CalcF(x[i])
#    
#fig = plt.figure(figsize=(6,12), dpi = 100)
#
#ax_x = fig.add_subplot(311)
#ax_x.set_ylabel("x")
#ax_x.plot(t, x[:,0])
#ax_x.axhline(color='r', ls='--')
#
#ax_y = fig.add_subplot(312)
#ax_y.set_ylabel("theta")
#ax_y.plot(t, x[:,1])
#ax_y.axhline(color='r', ls='--')

#%% initilization
h = 0.01 # time step.
N = 500 # horizon

n = 4 # number of states
m = 1 # number of inputs

# derivatives
Qx = np.zeros((N, n))
Qxx = np.zeros((N, n, n))
Qu = np.zeros((N, m))
Quu = np.zeros((N, m, m))
Qux = np.zeros((N, m, n))

# desired fixed point
xd = [0,np.pi,0,0]

# terminal cost = 1/2*(x-xd)'*QN*(x-xd)
QN = np.diag([1, 10, 1, 10])
# l(x,u) = 1/2*((x-xd)'*Q*(x-xd) + u'*R*u)
Q = np.diag([1, 10, 1, 10]) # lqr cost
R = np.eye(1) # lqr cost

delta_V = np.zeros(N+1)
Vx = np.zeros((N+1, n))
Vxx = np.zeros((N+1, n, n))

k = np.zeros((N, m))
K = np.zeros((N, n))

#%% iLQR
# initial trajectory 
x0 = np.array([0.,0,0,0])
x = np.zeros((N+1, n))
x[0] = x0
u = np.full((N, m), 0.1)
for t in range(N):
    x[t+1] = x[t] + h*CalcF(x[t],u[t])
x_new = np.zeros((N+1, n))
u_new = np.zeros((N, m))

# boundary conditions
Vxx[N] = QN 
Vx[N] = QN.dot(x[N]-xd)

# logging
Ni = 6 # number of iterations
Quu_inv_log = np.zeros((Ni, N, m, m))
# It really should be a while loop, but for linear systems one iteration seems 
# to be sufficient. And I am sure this can be proven. 
for j in range(Ni):
    if j > 0:
        x = x_new
        u = u_new
        Vx[N] = QN.dot(x[N]-xd)
        
    # backward pass
    for i in range(N-1, -1, -1): # i = N-1, ...., 0
        lx = Q.dot(x[i]-xd)
        lu = R.dot(u[i])
        lxx = Q
        luu = R
        fx = CalcFx(x[t], u[t])
        fu = CalcFu(x[t], u[t])
        
        Qx[i] = lx + fx.T.dot(Vx[i+1])
        Qu[i] = lu + fu.T.dot(Vx[i+1])
        Qxx[i] = lxx + fx.T.dot(Vxx[i+1].dot(fx))
        Quu[i] = luu + fu.T.dot(Vxx[i+1].dot(fu))
        Qux[i] = fu.T.dot(Vxx[i+1].dot(fx))
        
        # update derivatives of V
        Quu_inv = LA.inv(Quu[i])
        Quu_inv_log[j, i] = Quu_inv
        delta_V[i] = -0.5*Qu[i].dot(Quu_inv.dot(Qu[i]))
        Vx[i] = Qx[i] - Qu[i].dot(Quu_inv.dot(Qux[i]))
        Vxx[i] = Qxx[i] - Qux[i].T.dot(Quu_inv.dot(Qux[i]))
        
        # compute k and K
        k[i] = -Quu_inv.dot(Qu[i])
        K[i] = -Quu_inv.dot(Qux[i])
        
    # forward pass
    x_new[0] = x[0]
    for t in range(N):
        u_new[t] = u[t] + k[t] + K[t].dot(x_new[t] - x[t])
        x_new[t+1] = x_new[t] + h*CalcF(x_new[t],u_new[t])
    
    
#%% plot
t = np.array([i*h for i in range(N+1)])
fig = plt.figure(figsize=(6,12), dpi = 100)

ax_x = fig.add_subplot(311)
ax_x.set_ylabel("x")
ax_x.plot(t, x_new[:,0])
ax_x.axhline(color='r', ls='--')

ax_y = fig.add_subplot(312)
ax_y.set_ylabel("theta")
ax_y.plot(t, x_new[:,1])
ax_y.axhline(color='r', ls='--')

ax_u = fig.add_subplot(313)
ax_u.set_ylabel("u")
ax_u.set_xlabel("t")
ax_u.plot(t[0:-1], u_new)
ax_u.axhline(color='r', ls='--')




























