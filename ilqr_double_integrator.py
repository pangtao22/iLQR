import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from pydrake.all import LinearQuadraticRegulator
# Notations in this code follow "Synthesis and stabilization of complex 
# behaviors through online trajectory optimization" by Y. Tassa and E. Todorov.

#%% initilization
h = 0.01 # time step.
N = 100 # horizon

# discrete double integrator dynamics
A = np.array([[1, h], [0, 1]]) 
B = np.array([[h**2/2], [h]])
n = 2 # number of states
m = 1 # number of inputs

# derivatives
Qx = np.zeros((N, n))
Qxx = np.zeros((N, n, n))
Qu = np.zeros((N, m))
Quu = np.zeros((N, m, m))
Qux = np.zeros((N, m, n))

# l(x,u) = 1/2*(x'*Q*x + u'*R*u)
Q = np.diag([1, 1]) # lqr cost
R = np.eye(1) # lqr cost
# terminal cost = 1/2*x'*QN*x
K0, QN = LinearQuadraticRegulator(A,B,Q,R)

delta_V = np.zeros(N+1)
Vx = np.zeros((N+1, n))
Vxx = np.zeros((N+1, n, n))

k = np.zeros((N, m))
K = np.zeros((N, n))

#%% iLQR
# trajectory 
x0 = np.array([1,0])
x = np.zeros((N+1, n))
x[0] = x0
u = np.full((N, m), -1)
for t in range(N):
    x[t+1] = A.dot(x[t]) + B.dot(u[t])
x_new = np.zeros((N+1, n))
u_new = np.zeros((N, m))

# boundary conditions
Vxx[N] = QN 
Vx[N] = QN.dot(x[N])

# It really should be a while loop, but for linear systems one iteration seems 
# to be sufficient. And I am sure this can be proven. 
for j in range(1):
    if j > 0:
        x = x_new
        u = u_new
        Vx[N] = QN.dot(x[N])
        
    # backward pass
    for i in range(N-1, -1, -1): # i = N-1, ...., 0
        lx = Q.dot(x[i])
        lu = R.dot(u[i])
        lxx = Q
        luu = R
        fx = A
        fu = B
        Qx[i] = lx + fx.T.dot(Vx[i+1])
        Qu[i] = lu + fu.T.dot(Vx[i+1])
        Qxx[i] = lxx + fx.T.dot(Vxx[i+1].dot(fx))
        Quu[i] = luu + fu.T.dot(Vxx[i+1].dot(fu))
        Qux[i] = fu.T.dot(Vxx[i+1].dot(fx))
        
        # update derivatives of V
        Quu_inv = LA.inv(Quu[i])
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
        x_new[t+1] = A.dot(x_new[t]) + B.dot(u_new[t])
    
    
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




























