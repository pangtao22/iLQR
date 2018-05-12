import julia
import numpy as np
julia.Julia().include('quadrotor_dynamics.jl')
from julia import QuadrotorDynamics as QD

#%%
jul.include('MyModule.jl')
from julia import ABC

#%%
a = [1,2]
ABC.f_b(a)

#%%


#%%
QD.CalcPartials(np.zeros(16))