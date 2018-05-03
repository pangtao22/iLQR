from pydrake.forwarddiff import jacobian
from pydrake.all import DiscreteTimeLinearQuadraticRegulator
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
# Notations in this code follow "Synthesis and stabilization of complex 
# behaviors th

class WayPoint:
  def __init__(self, x, t, W, rho):
    x = np.asarray(x)
    assert len(x.shape) == 1 # x is 1d np array
    n = x.size
    assert W.shape == (n,n)
    assert t>=0
    assert rho > 0
    self.x = x
    self.t = t
    self.W = W
    self.rho = rho
    
class TrajectorySpecs:
  def __init__(self, x0, u0, xd, ud, h, N, Q, R, QN = None, xw_list=None):
    self.x0 = x0
    self.u0 = u0
    self.xd = xd
    self.ud = ud
    self.h = h
    self.N = N
    self.Q = Q
    self.R = R
    self.QN = QN
    self.xw_list = xw_list
    

class DiscreteTimeIterativeLQR:
  def __init__(self, CalcF, n, m):
    self.CalcF = CalcF # dynamics
    self.n = n # number of states
    self.m = m # number of inputs
    self.traj_specs = None # to be initialized in CalcTrajectory method.
  
  def PlotCosts(self, x, u, xd, ud, Q, R, QN, xw_list, h):
    N = u.shape[0]
    t = np.array([i*h for i in range(N)])
    fig = plt.figure(figsize=(6,12), dpi = 100)
    ax_x_lqr = fig.add_subplot(311)
    ax_x_lqr.set_ylabel('x_lqr_cost')
    ax_x_wpt = fig.add_subplot(312)
    ax_u_lqr = fig.add_subplot(313)
    ax_u_lqr.set_ylabel('u_lqr_cost')
    ax_u_lqr.set_xlabel('time(s)')
    
    # only plots the cost of the first waypoint.
    xw = xw_list[0]
    l_lqr_x = np.zeros(N)
    l_lqr_u = np.zeros(N)
    l_wpt = np.zeros(N)
    final_cost = (x[N] - xd).dot(QN.dot(x[N] - xd))
    for i in range(N):
      l_lqr_x[i] = (x[i] - xd).dot(Q.dot(x[i] - xd))
      l_lqr_u[i] = (u[i]-ud).dot(R.dot(u[i]-ud))  
      dx = x[i] - xw.x
      l_wpt[i] = dx.dot(xw.W.dot(dx))*self.discount(xw, i)
      
    ax_x_lqr.plot(t, l_lqr_x)
    ax_x_lqr.axhline(final_cost, final_cost, color='r', ls='--')
    
    ax_x_wpt.plot(t, l_wpt, color='b')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_x_wpt.set_ylabel('x_waypoint_cost', color='b')
    ax_x_wpt.tick_params('y', colors='b')
    
    ax_x_wpt2 = ax_x_wpt.twinx()
    # ax_x_wpt2.plot(t, self.discount(xw, t), 'r')
    ax_x_wpt2.set_ylabel('discount', color='r')
    ax_x_wpt2.tick_params('y', colors='r')
    
    ax_u_lqr.plot(t, l_lqr_u)
  
    
  # xw is a WayPoint
  def discount(self, xw, i):
    t = i * self.traj_specs.h
    return np.sqrt(0.5*xw.rho/np.pi)*np.exp(-0.5*xw.rho*(t-xw.t)**2)
  
  def CalcLqrCost(self, x, u, i0):
    N = u.shape[0]
    assert(x.shape == (N+1, self.n))
    assert(u.shape == (N, self.m))
    J = 0
    for i in range(i0, N):
      dx = x[i]-self.traj_specs.xd
      du = u[i]-self.traj_specs.ud
      J += dx.dot(self.traj_specs.Q.dot(dx))
      J += du.dot(self.traj_specs.R.dot(du))   
    dx_N = x[N] - self.traj_specs.xd
    J += dx_N.dot(self.traj_specs.QN.dot(dx_N))
    return J
    
  def CalcWayPointsCost(self, x, i0):
    if self.traj_specs.xw_list is None:
      return 0.
    
    N = x.shape[0]-1
    assert(x.shape == (N+1, self.n))
    W = 0.
    
    for i in range(i0, N):
      for xw in self.traj_specs.xw_list:
        dx = x[i] - xw.x
        W += dx.dot(xw.W.dot(dx))*self.discount(xw, i)
    return W
  
  '''
  Calculates the cost-to-go J of a paricular trajectory (x[.], u[.])
  starting at time i0.
  '''  
  def CalcJ(self, x, u, i0=0):
    assert(x.shape == (self.traj_specs.N+1, self.n))
    assert(u.shape == (self.traj_specs.N, self.m))
    J = 0
    J += self.CalcLqrCost(x, u, i0)
    J += self.CalcWayPointsCost(x, i0)
    return J
  
  # h: time step of iLQR
  # N: horizon
  # xd: goal/final state (should've been called xg)
  # Ni: Number of iLQR iterations
  # l(x,u) = 1/2*((x-xd)'*Q*(x-xd) + u'*R*u)
  # xw: list of WayPoints
  # terminal cost = 1/2*(x-xd)'*QN*(x-xd)
  def CalcTrajectory(self, traj_specs, Ni):
    assert(traj_specs.xd.shape == (self.n,))
    assert(traj_specs.ud.shape == (self.m,))

    def CallLQR(x, u, Q, R):
      f_x_u = jacobian(self.CalcF, np.hstack((x, u)))
      A = traj_specs.h*f_x_u[:, 0:self.n] + np.eye(self.n)
      B = traj_specs.h*f_x_u[:, self.n:self.n+self.m]
      K, P = DiscreteTimeLinearQuadraticRegulator(A, B, Q, R)
      return K, P
    
    # linearize about target/goal position
    if traj_specs.QN is None:
        Kd, traj_specs.QN = CallLQR(traj_specs.xd, traj_specs.ud, \
                                   traj_specs.Q, traj_specs.R)
    self.traj_specs = traj_specs
    
    # calculates lx
    def CalcLx(xi, i):
      Lx = traj_specs.Q.dot(xi - traj_specs.xd)
      # add contribution from waypoint weights
      if not(traj_specs.xw_list is None):
        for xw in traj_specs.xw_list:
          Lx += xw.W.dot(xi - xw.x)*self.discount(xw, i)
      return Lx
        
    # calculates lxx
    def CalcLxx(i):
      Lxx = traj_specs.Q.copy()
      if not(traj_specs.xw_list is None):
        for xw in traj_specs.xw_list:
          Lxx += xw.W*self.discount(xw,i)
      return Lxx

    # allocate storage for derivatives
    Qx = np.zeros((traj_specs.N, self.n))
    Qxx = np.zeros((traj_specs.N, self.n, self.n))
    Qu = np.zeros((traj_specs.N, self.m))
    Quu = np.zeros((traj_specs.N, self.m, self.m))
    Qux = np.zeros((traj_specs.N, self.m, self.n))
    
    delta_V = np.zeros(traj_specs.N+1)
    Vx = np.zeros((traj_specs.N+1, self.n))
    Vxx = np.zeros((traj_specs.N+1, self.n, self.n))
    
    k = np.zeros((traj_specs.N, self.m))
    K = np.zeros((traj_specs.N, self.m, self.n))
    
    # storage for trajectories
    x = np.zeros((Ni+1, traj_specs.N+1, self.n))
    u = np.zeros((Ni+1, traj_specs.N, self.m))
    x[0,0] = traj_specs.x0
    
#     initialize first trajectory by 
#     simulating forward with LQR controller about x0.
#    K0, P0 = CallLQR(x0, u0, Q, R)
#    for i in range(N):
#      u[0, i] = -K0.dot(x[0, i]-x0) + u0
#      x_u = np.hstack((x[0, i], u[0, i]))
#      x[0, i+1] = x[0, i] + h*self.CalcF(x_u)
    
    '''
    initialize first trajectory by 
    simulating forward with LQR controller about xd.
    '''
    Kd, Qd = CallLQR(traj_specs.xd, traj_specs.ud, traj_specs.Q, traj_specs.R)
    for i in range(traj_specs.N):
      u[0, i] = -Kd.dot(x[0, i]-traj_specs.xd) + traj_specs.ud
      x_u = np.hstack((x[0, i], u[0, i]))
      x[0, i+1] = x[0, i] + traj_specs.h*self.CalcF(x_u)
    

    # logging
    # Quu_inv_log = np.zeros((Ni, N, m, m))
    J = np.zeros(Ni+1)
    J[0] = self.CalcJ(x[0], u[0])
    
    # It really should be a while loop, but for linear systems one 
    # iteration seems sufficient. And I am sure this can be proven.
    # And for quadrotors it usually takes less than 5 iterations 
    # to converge.
    for j in range(Ni):
      # initialize boundary conditions
      Vxx[traj_specs.N] = traj_specs.QN 
      Vx[traj_specs.N] = traj_specs.QN.dot(x[j, traj_specs.N] - traj_specs.xd)    
           
      # backward pass
      for i in range(traj_specs.N-1, -1, -1): # i = N-1, ....
        lx = CalcLx(x[j, i], i)
        lu = traj_specs.R.dot(u[j, i] - traj_specs.ud)
        lxx = CalcLxx(i)
        luu = traj_specs.R
        x_u = np.hstack((x[j,i], u[j,i]))
        f_x_u = jacobian(self.CalcF, x_u)
        fx = traj_specs.h*f_x_u[:, 0:self.n] + np.eye(self.n)
        fu = traj_specs.h*f_x_u[:, self.n:self.n+self.m]
        
        Qx[i] = lx + fx.T.dot(Vx[i+1])
        Qu[i] = lu + fu.T.dot(Vx[i+1])
        Qxx[i] = lxx + fx.T.dot(Vxx[i+1].dot(fx))
        Quu[i] = luu + fu.T.dot(Vxx[i+1].dot(fu))
        Qux[i] = fu.T.dot(Vxx[i+1].dot(fx))
        
        # compute k and K
        Quu_inv = LA.inv(Quu[i])
        k[i] = -Quu_inv.dot(Qu[i])
        K[i] = -Quu_inv.dot(Qux[i])
        
        # update derivatives of V
        # Quu_inv_log[j, i] = Quu_inv
        delta_V[i] = 0.5*Qu[i].dot(k[i])
        Vx[i] = Qx[i] + Qu[i].dot(K[i])
        Vxx[i] = Qxx[i] + Qux[i].T.dot(K[i])
                     
      # forward pass
      del i
      x[j+1, 0] = x[j, 0]
      alpha = 1
      line_search_count = 0
      while True:  
        for t in range(traj_specs.N):
          u[j+1, t] = u[j, t] + alpha*k[t] + K[t].dot(x[j+1, t] - x[j, t])
          x_u = np.hstack((x[j+1, t], u[j+1, t]))
          x[j+1, t+1] = x[j+1, t] + traj_specs.h*self.CalcF(x_u)
        
        J_new = self.CalcJ(x[j+1], u[j+1], 0)

        if J_new <=  J[j]:
          J[j+1] = J_new
          break
        elif line_search_count > 5:
#          x[j+1] = x[j]
#          u[j+1] = u[j]
#          J[j+1] = J[j]
          J[j+1] = J_new
          break
        else:
          alpha *= 0.5
          line_search_count += 1
          print line_search_count
                
    return x, u, J, traj_specs.QN, Vx, Vxx

    

    





































