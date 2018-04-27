from pydrake.forwarddiff import jacobian
from pydrake.all import LinearQuadraticRegulator
import numpy as np
from numpy import linalg as LA
from numpy import sin, cos
import matplotlib.pyplot as plt
# Notations in this code follow "Synthesis and stabilization of complex 
# behaviors through online trajectory optimization" by Y. Tassa and E. Todorov.

#%% dynamics and derivatives

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

#%% initilization
h = 0.01 # time step.
N = 1000 # horizon

# desired fixed point
xd = np.array([0,1,0,0,0,0])
ud = np.array([0.5, 0.5])

# linearize about desired fixed point
f_x_u = jacobian(CalcF, np.hstack((xd, ud)))
A0 = f_x_u[:, 0:n]
B0 = f_x_u[:, n:n+m]

# l(x,u) = 1/2*((x-xd)'*Q*(x-xd) + u'*R*u)
Q = 100*np.eye(n) # lqr cost
R = np.eye(m) # lqr cost
K0, S0 = LinearQuadraticRegulator(A0, B0, Q, R)
QN = S0 # terminal cost = 1/2*(x-xd)'*QN*(x-xd)

# derivatives
Qx = np.zeros((N, n))
Qxx = np.zeros((N, n, n))
Qu = np.zeros((N, m))
Quu = np.zeros((N, m, m))
Qux = np.zeros((N, m, n))


delta_V = np.zeros(N+1)
Vx = np.zeros((N+1, n))
Vxx = np.zeros((N+1, n, n))

k = np.zeros((N, m))
K = np.zeros((N, m, n))


#%% iLQR
# initial trajectory 
x0 = np.array([0, 0.2, 0.1, 0, 0, -0.2])
x = np.zeros((N+1, n))
u = np.zeros((N, m))
x[0] = x0

def CalcJ(x,u,xd,ud):
    assert(x.shape == (N+1, n))
    assert(u.shape == (N, m))
    J = 0
    for i in range(N):
        J += (x[i]-xd).dot(Q.dot(x[i]-xd))
        J += (u[i]-ud).dot(R.dot(u[i]-ud))
    J += (x[N]-xd).dot(QN.dot(x[N]-xd))
    return J

# simulate forward
for t in range(N):
    u[t] = -K0.dot(x[t]) + ud
    x_u = np.hstack((x[t], u[t]))
    x[t+1] = x[t] + h*CalcF(x_u)
  
x_new = np.zeros((N+1, n))
u_new = np.zeros((N, m))
 #%% 
# boundary conditions
Vxx[N] = QN 
Vx[N] = QN.dot(x[N]-xd)

# logging
Ni = 3 # number of iterations

J = np.zeros(Ni+1)
J[0] = CalcJ(x, u, xd, ud)

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
        fx = f_x_u[:, 0:n]
        fu = f_x_u[:, n:n+m]
        
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
        
        J_new = CalcJ(x_new, u_new, xd, ud)

        if J_new <=  J[j]:
            J[j+1] = J_new
            break
        elif iteration_count > 5:
            break
        else:
            alpha *= 0.5
            iteration_count += 1
            print iteration_count

    
#%% plot
t = np.array([i*h for i in range(N+1)])
fig = plt.figure(figsize=(6,16), dpi = 100)

ax_x = fig.add_subplot(411)
ax_x.set_ylabel("x")
ax_x.plot(t, x_new[:,0])
ax_x.plot(t, x[:,0])
ax_x.axhline(color='r', ls='--')

ax_y = fig.add_subplot(412)
ax_y.set_ylabel("y")
ax_y.plot(t, x_new[:,1])
ax_y.plot(t, x[:,1])
ax_y.axhline(color='r', ls='--')

ax_theta = fig.add_subplot(413)
ax_theta.set_ylabel("theta")
ax_theta.plot(t, x_new[:,2])
ax_theta.plot(t, x[:,2])
ax_theta.axhline(color='r', ls='--')

ax_u = fig.add_subplot(414)
ax_u.set_ylabel("u")
ax_u.set_xlabel("t")
ax_u.plot(t[0:-1], u[:,0])
ax_u.axhline(color='r', ls='--')


#%% simulate and plot
dt = 0.001
T = 20000
t = dt*np.arange(T+1)
x = np.zeros((T+1, n))
x[0] = [0, 0, 0.1, 0, 0, 0]

for i in range(T):
    x_u = np.hstack((x[i], -K0.dot(x[i]-xd) + ud))
    x[i+1] = x[i] + dt*CalcF(x_u)
    
fig = plt.figure(figsize=(6,12), dpi = 100)

ax_x = fig.add_subplot(311)
ax_x.set_ylabel("x")
ax_x.plot(t, x[:,0])
ax_x.axhline(color='r', ls='--')

ax_y = fig.add_subplot(312)
ax_y.set_ylabel("y")
ax_y.plot(t, x[:,1])
ax_y.axhline(color='r', ls='--')

ax_phase = fig.add_subplot(313)
ax_phase.set_ylabel("theta")
ax_phase.set_xlabel("t")
ax_phase.plot(t, x[:,2])
ax_phase.axhline(color='r', ls='--')




























