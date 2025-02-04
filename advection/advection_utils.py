import firedrake as fd
from firedrake.__future__ import interpolate
import numpy as np
from sys import exit

from firedrake.adjoint.fourdvar_reduced_functional import covariance_norm

np.set_printoptions(legacy='1.25', precision=6)


def timestepper(mesh, V, dt, u):
    qn = fd.Function(V, name="qn")
    qn1 = fd.Function(V, name="qn1")

    def mass(q, phi):
        return fd.inner(q, phi)*fd.dx

    def tendency(q, phi):
        uc = fd.as_vector([fd.Constant(u)])
        n = fd.FacetNormal(mesh)
        un = fd.Constant(0.5)*(fd.dot(uc, n) + abs(fd.dot(uc, n)))
        return (- q*fd.div(phi*uc)*fd.dx
                + fd.jump(phi)*fd.jump(un*q)*fd.dS)

    # midpoint rule
    q = fd.TrialFunction(V)
    phi = fd.TestFunction(V)

    qh = fd.Constant(0.5)*(q + qn)
    eqn = mass(q - qn, phi) + fd.Constant(dt)*tendency(qh, phi)

    stepper = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(
            fd.lhs(eqn), fd.rhs(eqn), qn1,
            constant_jacobian=True))

    return qn, qn1, stepper


def analytic_solution(mesh, u, t, mag=1.0, phase=0.0):
    x, = fd.SpatialCoordinate(mesh)
    return mag*fd.sin(2*fd.pi*((x + phase) - u*t))

B = 10.
R = 0.1
Q = 0.2*B

ensemble = fd.Ensemble(fd.COMM_WORLD, 1)

nx = 20
cfl = 1.6

u = 1.
dx = 1./nx
dt = cfl*dx/u

mesh = fd.PeriodicUnitIntervalMesh(nx, comm=ensemble.comm)

V = fd.FunctionSpace(mesh, "DG", 1)

qn, qn1, stepper = timestepper(mesh, V, dt, u)

# initial conditions
x, = fd.SpatialCoordinate(mesh)
ic = fd.Function(V).interpolate(analytic_solution(mesh, u, t=0))
qn.assign(ic)

# observation operator
observation_freq = 4
observation_n = 3

# we have an extra observation at the initial time
observation_times = [i*observation_freq*dt
                     for i in range(observation_n+1)]

observation_locations = [
    [x] for x in [0.13, 0.18, 0.34, 0.36, 0.49, 0.61, 0.72, 0.99]
]

observation_mesh = fd.VertexOnlyMesh(mesh, observation_locations)
Vobs = fd.FunctionSpace(observation_mesh, "DG", 0)


def H(x):
    return fd.assemble(interpolate(x, Vobs))


# ground truth
targets = [
    fd.Function(V).interpolate(analytic_solution(mesh, u, t))
    for t in observation_times
]

# take observations
y = [H(x) for x in targets]


def observation_error(i):
    def obs_err(x):
        err = fd.Function(Vobs)
        err.assign(H(x) - y[i])
        return err
    return obs_err


prior_mag = 0.9
prior_phase = 0.1

# background
background = fd.Function(V).interpolate(
    analytic_solution(mesh, u, 0, mag=prior_mag, phase=prior_phase))

# other values to evaluate reduced functional at
values = [
    fd.Function(V).interpolate(
        analytic_solution(mesh, u, t+0.1, mag=1.1, phase=-0.2))
    for t in observation_times
]

values1 = [
    fd.Function(V).interpolate(
        analytic_solution(mesh, u, t+0.3, mag=0.8, phase=0.3))
    for t in observation_times
]
