module QuadrotorDynamics

using Rotations
using StaticArrays
using ForwardDiff

const kF = 1.0
const kM = 0.0245
const l = 0.175/sqrt(2.0) # (length of one arm) / sqrt(2)
const n = 12 # 12 states
const m = 4 # 4 inputs
# intertial and gravitational constants
const mass = 0.5
const I = @SMatrix [0.0023  0  0;
     0  0.0023  0;
     0  0  0.0040]
const g = 10.

# rpy_d = Phi * pqr
# pqr is the angular velocity expressed in Body frame.
function CalcPhi(rpy::AbstractVector)
    roll = rpy[1]
    pitch = rpy[2]
    sr = sin(roll)
    cr = cos(roll)
    sp = sin(pitch)
    cp = cos(pitch)

    T = eltype(cp)
    Phi = @SMatrix [one(T) sr*sp/cp cr*sp/cp;
           zero(T)  cr  -sr;
           zero(T) sr/cp cr/cp]
    return Phi
end

function CalcF!(xdot::Vector, x_u::Vector{T}) where {T}
    x = x_u[1:n]
    u = x_u[n+1:n+m]
    length(xdot) == n || throw(DimensionMisMatch("xdot has wrong size"))

    I_inv = inv(I)
    uF = kF * u
    uM = kM * u
    Fg = @SVector [0., 0., -mass*g]
    F = @SVector [0., 0., sum(uF)]
    M = @SVector [l*(-uF[1] - uF[2] + uF[3] + uF[4]),
         l*(-uF[1] - uF[4] + uF[2] + uF[3]),
         - uM[1] + uM[2] - uM[3] + uM[4]]

    rpy = SVector(x[4], x[5], x[6])
    rpy_d = SVector(x[10], x[11], x[12])
    R_WB = RotZYX(rpy[3], rpy[2], rpy[1])

    rpy_dual = ForwardDiff.Dual.(rpy, rpy_d)
    Phi_dual = CalcPhi(rpy_dual)
    Phi = ForwardDiff.value.(Phi_dual)
    Phi_dt = (x -> ForwardDiff.partials(x)[1]).(Phi_dual)

    # translational acceleration in world frame
    xyz_dd = 1./mass*(R_WB*F + Fg)

    # pqr: angular velocity in body frame
#     Phi_inv = CalcPhiInv(rpy)
    pqr = Phi \ rpy_d
    pqr_d = I_inv*(M - cross(pqr, I*pqr))

#     rpy_d = Phi * pqr ==>
#     rpy_dd = Phi_d * pqr + Phi * pqr_d
#     Phi_d.size = (3,3,3): Phi_d[i,j] is the partial of Phi[i,j]
#         w.r.t rpy.

#     Phi_d = CalcPhiD(rpy)
#     Phi = CalcPhi(rpy)
#     Phi_dt = zeros(T, 3,3)
#     for i in 1:3
#         for j in 1:3
#             Phi_dt[i,j] = sum(Phi_d[i,j,:].*rpy_d)
#         end
#     end

    rpy_dd = Phi*pqr_d + Phi_dt*pqr

    xdot[1:6] = x[7:12]
    xdot[7:9] = xyz_dd
    xdot[10:12] = rpy_dd
    return xdot
end

function CalcF(x_u::Vector{T}) where {T}
    xdot = zeros(T, n)
    CalcF!(xdot, x_u)
    return xdot
end

x_u = zeros(n+m)
xdot = zeros(n)
config = ForwardDiff.JacobianConfig(CalcF, x_u)
result = DiffResults.JacobianResult(xdot, x_u)

function CalcPartials(x_u)
    ForwardDiff.jacobian!(result, CalcF, x_u, config)
    return DiffResults.jacobian(result)
end


end
