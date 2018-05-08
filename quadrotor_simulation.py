import matplotlib.pyplot as plt
from pydrake.all import (DiagramBuilder, SignalLogger, Simulator)
from pydrake.all import LinearQuadraticRegulator
import numpy as np
from pydrake.systems.framework import VectorSystem
from pydrake.forwarddiff import jacobian
from quadrotor3D import Quadrotor, n, m, mass, g, CalcF, PlotTraj, PlotTrajectoryMeshcat

import meshcat
#%% get LQR controller about goal point
# fixed point
xd = np.zeros(n)
ud = np.zeros(m)
ud[:] = mass * g / 4
x_u = np.hstack((xd, ud))
partials = jacobian(CalcF, x_u)
A0 = partials[:, 0:n]
B0 = partials[:, n:n+m]
Q = 10*np.eye(n)
R = np.eye(m)

K0, S0 = LinearQuadraticRegulator(A0, B0, Q, R)

#%% Build drake diagram system and simulate.
builder = DiagramBuilder()
quad = builder.AddSystem(Quadrotor())

# controller system
class QuadLqrController(VectorSystem):
    def __init__(self):
        VectorSystem.__init__(self, n, m)

    # u(t) = -K.dot(x(t)) ==> y(t) = -K.dot(u)
    def _DoCalcVectorOutput(self, context, u, x, y):
        y[:] = -K0.dot(u-xd) + ud


# Create a simple block diagram containing our system.
controller = builder.AddSystem(QuadLqrController())
logger = builder.AddSystem(SignalLogger(n))

builder.Connect(controller.get_output_port(0), quad.get_input_port(0))
builder.Connect(quad.get_output_port(0), logger.get_input_port(0))
builder.Connect(quad.get_output_port(0), controller.get_input_port(0))
diagram = builder.Build()

# Create the simulator.
simulator = Simulator(diagram)

# Set the initial conditions, x(0).
state = simulator.get_mutable_context().get_mutable_continuous_state_vector()
x0 = np.zeros(n)
x0[0:3] = 0.5
x0[5] = np.pi/2
state.SetFromVector(x0)

# Simulate
simulator.StepTo(5.0)

#%% plot
PlotTraj(logger.data().T, None, None, logger.sample_times())

#%% open meshcat 
vis = meshcat.Visualizer()
vis.open()

#%% meshcat animation
PlotTrajectoryMeshcat(logger.data().T, vis, None, logger.sample_times())