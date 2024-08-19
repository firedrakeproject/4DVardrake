# Burgers equation N-wave on a 1D periodic domain
import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from firedrake.adjoint import (continue_annotation, pause_annotation,
                               get_working_tape, Control, minimize)
from firedrake.adjoint import ReducedFunctional  # noqa: F401
from firedrake.adjoint.all_at_once_reduced_functional import AllAtOnceReducedFunctional  # noqa: F401
from burgers_utils import noisy_double_sin, burgers_stepper

from functools import partial
import numpy as np
import argparse

pause_annotation()

Print = PETSc.Sys.Print


parser = argparse.ArgumentParser(
    description='Strong constraint 4DVar for the viscous Burgers equation using firedrake.AllAtOnceReducedFunctional.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=100, help='Number of elements.')
parser.add_argument('--cfl', type=float, default=1.0, help='Approximate Courant number.')
parser.add_argument('--ubar', type=float, default=1.0, help='Average initial velocity.')
parser.add_argument('--tend', type=float, default=1.0, help='Final integration time.')
parser.add_argument('--re', type=float, default=1e2, help='Approximate Reynolds number.')
parser.add_argument('--theta', type=float, default=0.5, help='Implicit timestepping parameter.')
parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance of optimiser.')
parser.add_argument('--prior_mag', type=float, default=1.1, help='Magnitude of background vs truth.')
parser.add_argument('--prior_shift', type=float, default=0.05, help='Phase shift in background vs truth.')
parser.add_argument('--prior_noise', type=float, default=0.05, help='Noise magnitude in background.')
parser.add_argument('--B', type=float, default=1e-1, help='Background trust weighting.')
parser.add_argument('--R', type=float, default=1.0, help='Observation trust weighting.')
parser.add_argument('--obs_spacing', type=str, default='random', choices=['random', 'equidistant'], help='How observation points are distributed in space.')
parser.add_argument('--obs_freq', type=int, default=10, help='Frequency of observations in time.')
parser.add_argument('--obs_density', type=int, default=10, help='Frequency of observations in space. Only used if obs_spacing=equidistant.')
parser.add_argument('--n_obs', type=int, default=10, help='Number of observations in space. Only used if obs_spacing=random.')
parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
parser.add_argument('--taylor_test', action='store_true', help='Run adjoint Taylor tes and exitt.')
parser.add_argument('--vtk', action='store_true', help='Write out timeseries to VTK file.')
parser.add_argument('--progress', action='store_true', help='Show tape progress bar.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

Print("Setting up problem")
np.random.seed(args.seed)

# problem parameters
nu = 1/args.re
dt = args.cfl/(args.nx*args.ubar)
theta = args.theta

# error covariance inverse
B = fd.Constant(args.B)
R = fd.Constant(args.R)

mesh = fd.PeriodicUnitIntervalMesh(args.nx)
mesh_out = fd.UnitIntervalMesh(args.nx)

un, un1, stepper = burgers_stepper(nu, dt, theta, mesh)
V = un.function_space()

# true initial condition and perturbed starting guess
ic_target = noisy_double_sin(V, avg=args.ubar, mag=0.5, shift=-0.02, noise=None)
background = noisy_double_sin(V, avg=args.ubar, mag=0.5*args.prior_mag,
                              shift=args.prior_shift, noise=args.prior_noise, seed=args.seed)

uend_target = fd.Function(V, name="Target")

un.assign(ic_target)
un1.assign(ic_target)

# target forward solution

Print("Running target forward model")

# observations on VOM
if args.obs_spacing == 'equidistant':
    coords = mesh_out.coordinates.dat.data
    obs_points = [[coords[i]] for i in range(0, len(coords), args.obs_density)]
if args.obs_spacing == 'random':
    obs_points = [[x] for x in sorted(np.random.random_sample(args.n_obs))]
# Print(f"{obs_points = }")

obs_mesh = fd.VertexOnlyMesh(mesh, obs_points)
Vobs = fd.VectorFunctionSpace(obs_mesh, "DG", 0)


def H(x):
    return fd.assemble(interpolate(x, Vobs))


y = []
utargets = [ic_target.copy(deepcopy=True)]

# calculate "ground truth" values from target ic

t = 0.0
nsteps = int(0)
obs_times = [0]
y.append(H(un))
while ((t + 0.5*dt) <= args.tend):
    stepper.solve()
    un.assign(un1)
    t += dt
    utargets.append(un.copy(deepcopy=True))
    if ((nsteps+1) % args.obs_freq) == 0:
        obs_times.append(nsteps)
        y.append(H(un))
    nsteps += int(1)
Print(f"Number of timesteps {nsteps = }")
Print(f"Number of observations {len(y) = }")

Print("Setting up adjoint model")


# weighted l2 inner product for error covariances
def wl2prod(x, w=1.0):
    return fd.assemble(fd.inner(x, w*x)*fd.dx)


background_iprod = partial(wl2prod, w=B)
observation_iprod = partial(wl2prod, w=R)


# Initialise forward model from prior/background initial conditions
# and log observations as we go
continue_annotation()
tape = get_working_tape()

un.assign(background)
un1.assign(background)

uapprox = [background.copy(deepcopy=True, annotate=False)]


def observation_err(i, state):
    return H(state) - y[i]


Jhat = AllAtOnceReducedFunctional(Control(background),
                                  observation_err=partial(observation_err, 0),
                                  observation_iprod=observation_iprod,
                                  background_iprod=background_iprod,
                                  weak_constraint=False)

Print("Running forward model")
observation_idx = 1
for i in range(nsteps):
    stepper.solve()
    un.assign(un1)
    uapprox.append(un.copy(deepcopy=True, annotate=False))

    if i == obs_times[observation_idx]:
        Jhat.set_observation(un, partial(observation_err, observation_idx),
                             observation_iprod=observation_iprod)
        observation_idx += 1


if args.taylor_test:
    from firedrake.adjoint import taylor_test
    from sys import exit
    Print("Running Taylor test on strong-constraint reduced functional")
    h = fd.Function(V)
    h.dat.data[:] = np.random.random_sample(h.dat.data.shape)
    Print(f"{taylor_test(Jhat, background, h) = }")
    exit()

Print("Minimizing 4DVar functional")
if args.progress:
    tape.progress_bar = fd.ProgressBar

options = {'disp': True, 'maxcor': 30, 'gtol': args.tol}
ic_opt = minimize(Jhat, options=options, method="L-BFGS-B")

Print(f"Initial functional: {Jhat(background)}")
Print(f"Final functional: {Jhat(ic_opt)}")

tape.clear_tape()
pause_annotation()

uopt = [ic_opt.copy(deepcopy=True)]

# calculate timeseries from optimised initial conditions
un.assign(ic_opt)
un1.assign(ic_opt)
for _ in range(nsteps):
    stepper.solve()
    un.assign(un1)
    uopt.append(un.copy(deepcopy=True))

Print(f"Initial ic error: {fd.errornorm(background, ic_target) = }")
Print(f"Final ic error: {fd.errornorm(ic_opt, ic_target) = }")
Print(f"Initial terminal error: {fd.errornorm(uapprox[-1], utargets[-1]) = }")
Print(f"Final terminal error: {fd.errornorm(uopt[-1], utargets[-1]) = }")

if args.vtk:
    from burgers_utils import InterpWriter
    # output on non-periodic mesh
    V_out = fd.VectorFunctionSpace(mesh_out, "CG", 1)
    vnames = ["TargetVelocity", "InitialGuess", "OptimisedVelocity"]
    write = InterpWriter("output/burgers_target.pvd", V, V_out, vnames).write
    for i, us in enumerate(zip(utargets, uapprox, uopt)):
        write(*us, t=i*dt)

tape.clear_tape()
