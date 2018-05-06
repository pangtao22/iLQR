import matplotlib.pyplot as plt
from pydrake.all import (DiagramBuilder, SignalLogger, Simulator)
import quadrotor_3D_drake_system as quad
from pydrake.all import LinearQuadraticRegulator
import numpy as np
from pydrake.systems.framework import VectorSystem
from pydrake.forwarddiff import jacobian

builder = DiagramBuilder()
system = builder.AddSystem(quad.Quadrotor())

# fixed point
n = quad.n
m = quad.m
xd = np.zeros(n)
ud = np.zeros(m)
ud[:] = system.mass * system.g / 4
x_u = np.hstack((xd, ud))
partials = jacobian(system.f, x_u)
A0 = partials[:, 0:n]
B0 = partials[:, n:n+m]
Q = 10*np.eye(n)
R = np.eye(m)

# get LQR controller about fixed point
K0, S0 = LinearQuadraticRegulator(A0, B0, Q, R)

#%%
# controller system
class QuadLqrController(VectorSystem):
    def __init__(self):
        VectorSystem.__init__(self, n, m)

    # u(t) = -K.dot(x(t)) ==> y(t) = -K.dot(u)
    def _DoCalcVectorOutput(self, context, u, x, y):
        y[:] = -K0.dot(u)


# Create a simple block diagram containing our system.
controller = builder.AddSystem(QuadLqrController())
logger = builder.AddSystem(SignalLogger(n))

builder.Connect(controller.get_output_port(0), system.get_input_port(0))
builder.Connect(system.get_output_port(0), logger.get_input_port(0))
builder.Connect(system.get_output_port(0), controller.get_input_port(0))
diagram = builder.Build()

# Create the simulator.
simulator = Simulator(diagram)

# Set the initial conditions, x(0).
state = simulator.get_mutable_context().get_mutable_continuous_state_vector()
x0 = np.zeros(n)
x0[0:3] = 0.1
state.SetFromVector(x0)

# Simulate
simulator.StepTo(5.0)