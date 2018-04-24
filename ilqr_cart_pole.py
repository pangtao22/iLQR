from pydrake.forwarddiff import jacobian
from pydrake.all import LinearQuadraticRegulator
import numpy as np
from numpy import linalg as LA
from numpy import sin, cos
import matplotlib.pyplot as plt
# Notations in this code follow "Synthesis and stabilization of complex 
# behaviors through online trajectory optimization" by Y. Tassa and E. Todorov.

#%% dynamics and derivatives
# dynamics
def CalcF(x_u):
    assert(x_u.size == 5)
    x = x_u[0:4]
    u = x_u[4:5]

    theta = x[1]
    xc_dot = x[2]
    theta_dot = x[3]
    
    
    xc_dotdot = u[0] + theta_dot**2*sin(theta) + sin(theta)*cos(theta)
    xc_dotdot /= 1+sin(theta)**2
    
    theta_dotdot = -cos(theta)*u[0] - -theta_dot**2*sin(2*theta)/2 - 2*sin(theta)
    theta_dotdot /= 1+sin(theta)**2
    
    # damping
    # x_dot[2] += -0.1*xc_dot
    # x_dot[3] += -0.1*theta_dot
    
    return np.array([xc_dot, theta_dot, xc_dotdot, theta_dotdot])

# x derivatives
def CalcFx(x_u):
    assert(x_u.size == 5)
    x = x_u[0:4]
    u = x_u[4:5]
    
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
def CalcFu(x_u):
    assert(x_u.size == 5)
    x = x_u[0:4]
    u = x_u[4:5]
    
    theta = x[1]   
    fu = np.array([0,0,1,-cos(theta)]) / (1+sin(theta)**2)
    fu.resize((4,1))
    
    return fu

#%% compare autodiff and analytic derivatives
#    
#x_u = np.array([0,0,0,0,0])
#
#print CalcFx(x_u)
#print CalcFu(x_u)
#print jacobian(CalcF, x_u)

#%% simulate and plot
dt = 0.001
T = 20000
t = dt*np.arange(T+1)
x = np.zeros((T+1, 4))
x[0] = [0, np.pi-0.05, 0, 0]

K0, S0 = LinearQuadraticRegulator(A0, B0, 100*np.diag([0.1,1,1,1]), 1*np.eye(1))
for i in range(T):
    x_u = np.hstack((x[i], K0.dot(x[i]-xd)))
    x[i+1] = x[i] + dt*CalcF(x_u)
    
fig = plt.figure(figsize=(6,12), dpi = 100)

ax_x = fig.add_subplot(311)
ax_x.set_ylabel("x")
ax_x.plot(t, x[:,0])
ax_x.axhline(color='r', ls='--')

ax_y = fig.add_subplot(312)
ax_y.set_ylabel("theta")
ax_y.plot(t, x[:,1])
ax_y.axhline(color='r', ls='--')

#%% initilization
h = 0.01 # time step.
N = 400 # horizon

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
QN = 100*np.diag([1, 1, 1, 1])
# l(x,u) = 1/2*((x-xd)'*Q*(x-xd) + u'*R*u)
Q = 100*np.diag([1, 1, 1, 1]) # lqr cost
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
# get LQR controller about the upright fixed point.
f_x_u = jacobian(CalcF, np.hstack((xd, [0])))
A0 = f_x_u[:, 0:4]
B0 = f_x_u[:, 4:5]
K0, S0 = LinearQuadraticRegulator(A0, B0, Q, R)

# simulate forward
for t in range(N):
    x_u = np.hstack((x[t], K0.dot(x[t])))
    x[t+1] = x[t] + h*CalcF(x_u)
    
x_new = np.zeros((N+1, n))
u_new = np.zeros((N, m))

# boundary conditions
Vxx[N] = QN 
Vx[N] = QN.dot(x[N]-xd)

# logging
Ni = 5 # number of iterations
Quu_inv_log = np.zeros((Ni, N, m, m))
# It really should be a while loop, but for linear systems one iteration seems 
# to be sufficient. And I am sure this can be proven. 
for j in range(Ni):
    if j > 0:
        x = x_new
        u = u_new
        Vx[N] = QN.dot(x[N]-xd)
        
    del t   
    # backward pass
    for i in range(N-1, -1, -1): # i = N-1, ...., 0
        lx = Q.dot(x[i]-xd)
        lu = R.dot(u[i])
        lxx = Q
        luu = R
        x_u = np.hstack((x[i], u[i]))
        f_x_u = jacobian(CalcF, x_u)
        fx = f_x_u[:, 0:4]
        fu = f_x_u[:, 4:5]
#        fx = CalcFx(x_u)
#        fu = CalcFu(x_u)
        
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
    del i
    x_new[0] = x[0]
    for t in range(N):
        u_new[t] = u[t] + k[t] + K[t].dot(x_new[t] - x[t])
        x_u_new = np.hstack((x_new[t], u_new[t]))
        x_new[t+1] = x_new[t] + h*CalcF(x_u_new)
    
    
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
ax_y.axhline(np.pi, color='r', ls='--')

ax_u = fig.add_subplot(313)
ax_u.set_ylabel("u")
ax_u.set_xlabel("t")
ax_u.plot(t[0:-1], u_new)
ax_u.axhline(color='r', ls='--')




























