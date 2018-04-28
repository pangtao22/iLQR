from pydrake.forwarddiff import jacobian
from pydrake.all import LinearQuadraticRegulator
from pydrake.all import DiscreteTimeLinearQuadraticRegulator
import numpy as np
from numpy import linalg as LA
# Notations in this code follow "Synthesis and stabilization of complex 
# behaviors through online trajectory optimization" by Y. Tassa and E. Todorov.

class DiscreteTimeIterativeLQR:
    def __init__(self, CalcF, n, m):
        self.CalcF = CalcF # dynamics
        self.n = n # number of states
        self.m = m # number of inputs

    
    # h: time step of iLQR
    # N: horizon
    # xd: desired fixed point/final state
    # Ni: Number of iLQR iterations
    # l(x,u) = 1/2*((x-xd)'*Q*(x-xd) + u'*R*u)
    def CalcTrajectory(self, x0, u0, xd, ud, h, N, Q, R, Ni):
        assert(xd.shape == (self.n,))
        assert(ud.shape == (self.m,))
        n = self.n
        m = self.m
        
        # linearize about initial position
        f_x_u = jacobian(self.CalcF, np.hstack((x0, u0)))
        A0 = h*f_x_u[:, 0:n] + np.eye(n)
        B0 = h*f_x_u[:, n:n+m]
        # terminal cost = 1/2*(x-xd)'*QN*(x-xd)
        K0, QN = DiscreteTimeLinearQuadraticRegulator(A0, B0, Q, R)
        
        # allocate storage for derivatives
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
        
        # storage for trajectories
        x = np.zeros((Ni+1, N+1, n))
        u = np.zeros((Ni+1, N, m))
        x[0,0] = x0

        # initialize first trajectory by 
        # simulating forward with LQR controller about x0.
        for i in range(N):
            u[0, i] = -K0.dot(x[0, i]-x0) + u0
            x_u = np.hstack((x[0, i], u[0, i]))
            x[0, i+1] = x[0, i] + h*self.CalcF(x_u)
            
        # Calculates the cost-to-go J of a paricular trajectory (x[i], u[i])
        def CalcJ(x, u):
            assert(x.shape == (N+1, n))
            assert(u.shape == (N, m))
            J = 0
            for i in range(N):
                J += (x[i]-xd).dot(Q.dot(x[i]-xd))
                J += (u[i]-ud).dot(R.dot(u[i]-ud))
            J += (x[N]-xd).dot(QN.dot(x[N]-xd))
            return J
        
        # logging
        Quu_inv_log = np.zeros((Ni, N, m, m))
        J = np.zeros(Ni+1)
        J[0] = CalcJ(x[0], u[0])
        
        # It really should be a while loop, but for linear systems one 
        # iteration seems sufficient. And I am sure this can be proven.
        # And for quadrotors it usually takes less than 5 iterations 
        # to converge.
        for j in range(Ni):
            # initialize boundary conditions
            Vxx[N] = QN 
            Vx[N] = QN.dot(x[j, N]-xd)    
                 
            # backward pass
            for i in range(N-1, -1, -1): # i = N-1, ...., 0
                lx = Q.dot(x[j, i] - xd)
                lu = R.dot(u[j, i] - ud)
                lxx = Q
                luu = R
                x_u = np.hstack((x[j,i], u[j,i]))
                f_x_u = jacobian(self.CalcF, x_u)
                fx = h*f_x_u[:, 0:n] + np.eye(n)
                fu = h*f_x_u[:, n:n+m]
                
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
            x[j+1, 0] = x[j, 0]
            alpha = 1
            line_search_count = 0
            while True:  
                for t in range(N):
                    u[j+1, t] = u[j, t] + alpha*k[t] + K[t].dot(x[j+1, t] - x[j, t])
                    x_u = np.hstack((x[j+1, t], u[j+1, t]))
                    x[j+1, t+1] = x[j+1, t] + h*self.CalcF(x_u)
                
                J_new = CalcJ(x[j+1], u[j+1])
        
                if J_new <=  J[j]:
                    J[j+1] = J_new
                    break
                elif line_search_count > 5:
                    J[j+1] = J_new
                    break
                else:
                    alpha *= 0.5
                    line_search_count += 1
                    print line_search_count
                    
        return x, u, J, QN
    
        

        





































