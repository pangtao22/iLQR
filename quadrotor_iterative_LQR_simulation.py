import matplotlib.pyplot as plt
from pydrake.all import (DiagramBuilder, SignalLogger, Simulator, PortDataType,
                         LinearQuadraticRegulator, BasicVector)
import numpy as np
from pydrake.systems.framework import LeafSystem
from pydrake.forwarddiff import jacobian
from quadrotor3D import (Quadrotor, n, m, mass, g, CalcF, PlotTraj, PlotTrajectoryMeshcat)
from ilqr_quadrotor_3D import traj_specs, planner
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
class QuadLqrController(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self._DeclareInputPort(PortDataType.kVectorValued, n)
        self._DeclareVectorOutputPort(BasicVector(m), self._DoCalcVectorOutput)
        self._DeclareDiscreteState(m) # state of the controller system is u
        self._DeclarePeriodicDiscreteUpdate(period_sec=0.005) # update u at 200Hz


    # u(t) = -K.dot(x(t)) ==> y(t) = -K.dot(u)
    def ComputeControlInput(self, x, t):
        return -K0.dot(x-xd) + ud


    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem._DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        new_control_input = discrete_state.get_mutable_vector().get_mutable_value()
        x = self.EvalVectorInput(context, 0).get_value()
        new_u = self.ComputeControlInput(x, context.get_time())
        new_control_input[:] = new_u


    def _DoCalcVectorOutput(self, context, y_basic_vector):
        control_output = context.get_discrete_state_vector().get_value()
        y = y_basic_vector.get_mutable_value()
        y[:] = control_output


# Create a simple block diagram containing our system.
controller = builder.AddSystem(QuadLqrController())
logger_x = builder.AddSystem(SignalLogger(n))
logger_u = builder.AddSystem(SignalLogger(m))

builder.Connect(controller.get_output_port(0), quad.get_input_port(0))
builder.Connect(quad.get_output_port(0), logger_x.get_input_port(0))
builder.Connect(quad.get_output_port(0), controller.get_input_port(0))
builder.Connect(controller.get_output_port(0), logger_u.get_input_port(0))
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
PlotTraj(logger_x.data().T, None, None, logger_x.sample_times())

#%% open meshcat 
vis = meshcat.Visualizer()
vis.open()

#%% meshcat animation
PlotTrajectoryMeshcat(logger_x.data().T, vis, None, logger_x.sample_times())