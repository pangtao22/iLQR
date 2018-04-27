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
    assert(x_u.size == 3)
    theta = x_u[0]
    theta_dot = x_u[1]
    u = x_u[2]    
    return np.array([theta_dot, u - sin(theta)])

# energy shaping controller
def Tau(x):
    theta = x[0]
    theta_dot = x[1]
    
    E_desired = 1
    E = 0.5*theta_dot**2 - cos(theta)
    E_tilt = E - E_desired
    
    return np.array([-theta_dot*E_tilt])

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
x = np.zeros((T+1, 2))
x[0] = [0, 0.1]

# desired fixed point
xd = np.array([-np.pi, 0])

# linearize about upright fixed point
f_x_u = jacobian(CalcF, np.hstack((xd, [0])))
A0 = f_x_u[:, 0:2]
B0 = f_x_u[:, 2:3]

K0, S0 = LinearQuadraticRegulator(A0, B0, 10*np.diag([1,1]), 1*np.eye(1))
for i in range(T):
    x_u = np.hstack((x[i], Tau(x[i])))
    x[i+1] = x[i] + dt*CalcF(x_u)
    
fig = plt.figure(figsize=(6,12), dpi = 100)

ax_x = fig.add_subplot(311)
ax_x.set_ylabel("x")
ax_x.plot(t, x[:,0])
ax_x.axhline(np.pi, color='r', ls='--')

ax_y = fig.add_subplot(312)
ax_y.set_ylabel("theta")
ax_y.plot(t, x[:,1])
ax_y.axhline(color='r', ls='--')

ax_phase = fig.add_subplot(313)
ax_phase.set_ylabel("theta_dot")
ax_phase.set_xlabel("theta")
ax_phase.plot(x[:,0], x[:,1])
ax_phase.axhline(color='r', ls='--')

#%% initilization
h = 0.01 # time step.
N = 400 # horizon

n = 2 # number of states
m = 1 # number of inputs

# derivatives
Qx = np.zeros((N, n))
Qxx = np.zeros((N, n, n))
Qu = np.zeros((N, m))
Quu = np.zeros((N, m, m))
Qux = np.zeros((N, m, n))


# terminal cost = 1/2*(x-xd)'*QN*(x-xd)
QN = 100*np.diag([1, 1])
# l(x,u) = 1/2*((x-xd)'*Q*(x-xd) + u'*R*u)
Q = QN # lqr cost
R = np.eye(1) # lqr cost

delta_V = np.zeros(N+1)
Vx = np.zeros((N+1, n))
Vxx = np.zeros((N+1, n, n))

k = np.zeros((N, m))
K = np.zeros((N, n))


#%% iLQR
# initial trajectory 
x0 = np.array([0., 0.1])
x = np.zeros((N+1, n))
u = np.zeros((N, m))
x[0] = x0

def CalcJ(x,u):
    assert(x.shape == (N+1, n))
    assert(u.shape == (N, m))
    
    J = 0
    for i in range(N):
        J += x[i].dot(Q.dot(x[i])) + u[i].dot(R.dot(u[i]))
        
    J += x[N].dot(QN.dot(x[N]))
    
    return J

# simulate forward
for t in range(N):
    u[t] = Tau(x[t])
    x_u = np.hstack((x[t], u[t]))
    x[t+1] = x[t] + h*CalcF(x_u)
    
x_new = np.zeros((N+1, n))
u_new = np.zeros((N, m))


# boundary conditions
Vxx[N] = QN 
Vx[N] = QN.dot(x[N]-xd)

# logging
Ni = 20 # number of iterations

J = np.zeros(Ni+1)
J[0] = CalcJ(x, u)

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
        fx = f_x_u[:, 0:2]
        fu = f_x_u[:, 2:3]
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
    alpha = 1
    iteration_count = 0
    while True:  
        for t in range(N):
            u_new[t] = u[t] + alpha*k[t] + K[t].dot(x_new[t] - x[t])
            x_u_new = np.hstack((x_new[t], u_new[t]))
            x_new[t+1] = x_new[t] + h*CalcF(x_u_new)
        
        J_new = CalcJ(x_new, u_new)
        
        if J_new < J[j]:
            J[j+1] = J_new
            break
        else:
            alpha *= 0.8
            iteration_count += 1
            print iteration_count
         
    
    
#%% plot
t = np.array([i*h for i in range(N+1)])
fig = plt.figure(figsize=(6,12), dpi = 100)

ax_x = fig.add_subplot(311)
ax_x.set_ylabel("theta")
ax_x.plot(t, x_new[:,0])
ax_x.axhline(np.pi, color='r', ls='--')

ax_y = fig.add_subplot(312)
ax_y.set_ylabel("theta_dot")
ax_y.plot(t, x_new[:,1])
ax_y.axhline(color='r', ls='--')

ax_u = fig.add_subplot(313)
ax_u.set_ylabel("u")
ax_u.set_xlabel("t")
ax_u.plot(t[0:-1], u_new)
ax_u.axhline(color='r', ls='--')




























