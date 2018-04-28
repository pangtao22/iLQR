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
    def CalcTrajectory(self, x0 , xd, ud, h, N, Q, R, Ni):
        assert(xd.shape == (self.n,))
        assert(ud.shape == (self.m,))
        n = self.n
        m = self.m
        
        # linearize about desired fixed point
        f_x_u = jacobian(self.CalcF, np.hstack((xd, ud)))
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
        x = np.zeros((N+1, n))
        x[0] = x0
        u = np.zeros((N, m))

        # simulate forward with LQR controller about x0.
        for i in range(N):
            u[i] = -K0.dot(x[i]-xd) + ud
            #-----------------hack--------------------------
#            u[i] = -0.1
            #-----------------------------------------------
            x_u = np.hstack((x[i], u[i]))
            x[i+1] = x[i] + h*self.CalcF(x_u)
            
        # Calculates the cost-to-go J of a paricular trajectory (x,u)
        def CalcJ(x, u):
            assert(x.shape == (N+1, n))
            assert(u.shape == (N, m))
            J = 0
            for i in range(N):
                J += (x[i]-xd).dot(Q.dot(x[i]-xd))
                J += (u[i]-ud).dot(R.dot(u[i]-ud))
            J += (x[N]-xd).dot(QN.dot(x[N]-xd))
            return J
        
        # boundary conditions
        Vxx[N] = QN 
        Vx[N] = QN.dot(x[N]-xd)        
         
        # logging
        Quu_inv_log = np.zeros((Ni, N, m, m))
        J = np.zeros(Ni+1)
        J[0] = CalcJ(x, u)
        
        # It really should be a while loop, but for linear systems one 
        # iteration seems sufficient. And I am sure this can be proven. 
        x_new = np.zeros((N+1, n))
        u_new = np.zeros((N, m))
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
                x_u = np.hstack((x[i], u[i]))
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
            x_new[0] = x[0]
            alpha = 1
            line_search_count = 0
            while True:  
                for t in range(N):
                    u_new[t] = u[t] + alpha*k[t] + K[t].dot(x_new[t] - x[t])
                    x_u_new = np.hstack((x_new[t], u_new[t]))
                    x_new[t+1] = x_new[t] + h*self.CalcF(x_u_new)
                
                J_new = CalcJ(x_new, u_new)
        
                if J_new <=  J[j]:
                    J[j+1] = J_new
                    break
                elif line_search_count > 5:
                    break
                else:
                    alpha *= 0.5
                    line_search_count += 1
                    print line_search_count
                    
            return x_new, u_new, x, u, J, QN
    
        

        





































