import numpy as np
from pydrake.all import (DiagramBuilder, SignalLogger, Simulator, PortDataType,
                         LinearQuadraticRegulator, BasicVector)
from pydrake.systems.framework import LeafSystem
from pydrake.forwarddiff import jacobian
from quadrotor3D import (Quadrotor, n, m, mass, g, CalcF, PlotTraj, PlotTrajectoryMeshcat)
from ilqr_quadrotor_3D import traj_specs, planner
# visualization
import matplotlib.pyplot as plt
import meshcat
#%% get iLQR controller
# fixed point
x_nominal, u_nominal, J, QN, Vx, Vxx, k, K = planner.CalcTrajectory(traj_specs, is_logging_trajectories = False)

PlotTraj(x_nominal, traj_specs.h, traj_specs.xw_list)

#%% Build drake diagram system and simulate.
builder = DiagramBuilder()
quad = builder.AddSystem(Quadrotor())

# controller system: applies iLQR controller to the quadcopter
class QuadLqrController(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self._DeclareInputPort(PortDataType.kVectorValued, n)
        self._DeclareVectorOutputPort(BasicVector(m), self._DoCalcVectorOutput)
        self._DeclareDiscreteState(m) # state of the controller system is u
        self._DeclarePeriodicDiscreteUpdate(period_sec=traj_specs.h) # update u every h seconds.

    # u(t) = -K.dot(x(t)) ==> y(t) = -K.dot(u)
    def ComputeControlInput(self, x, t):
        i = int(round(t/traj_specs.h))
        if i >= len(k):
            i = len(k) -1
        print i, t
        print 'u_nominal[i]:', u_nominal[i]
        print 'k[i]:', k[i]
        print 'K*x_error:', K[i].dot(x-x_nominal[i])
        return u_nominal[i] + K[i].dot(x-x_nominal[i])


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
state.SetFromVector(traj_specs.x0)

# Simulate
simulator.StepTo(traj_specs.h * traj_specs.N)

#%% plot
PlotTraj(logger_x.data().T, None, None, logger_x.sample_times())

#%% open meshcat 
vis = meshcat.Visualizer()
vis.open()

#%% meshcat animation
PlotTrajectoryMeshcat(logger_x.data().T, logger_x.sample_times(), vis)