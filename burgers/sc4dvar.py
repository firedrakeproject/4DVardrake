# Burgers equation N-wave on a 1D periodic domain
import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from firedrake.adjoint import (continue_annotation, get_working_tape, pause_annotation,
                               Control, ReducedFunctional, minimize)
from burgers_utils import noisy_double_sin, burgers_stepper
import numpy as np
from functools import partial
import argparse

np.set_printoptions(legacy='1.25')

Print = PETSc.Sys.Print

parser = argparse.ArgumentParser(
    description='Strong constraint 4DVar for the viscous Burgers equation.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=100, help='Number of elements.')
parser.add_argument('--cfl', type=float, default=1.0, help='Approximate Courant number.')
parser.add_argument('--ubar', type=float, default=1.0, help='Average initial velocity.')
parser.add_argument('--tend', type=float, default=1.0, help='Final integration time.')
parser.add_argument('--re', type=float, default=1e2, help='Approximate Reynolds number.')
parser.add_argument('--theta', type=float, default=0.5, help='Implicit timestepping parameter.')
parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance of optimiser.')
parser.add_argument('--prior_mag', type=float, default=1.05, help='Magnitude of background vs truth.')
parser.add_argument('--prior_shift', type=float, default=0.025, help='Phase shift in background vs truth.')
parser.add_argument('--prior_noise', type=float, default=0.025, help='Noise magnitude in background.')
parser.add_argument('--B', type=float, default=1e-1, help='Background trust weighting.')
parser.add_argument('--R', type=float, default=1e1, help='Observation trust weighting.')
parser.add_argument('--obs_freq', type=int, default=10, help='Frequency of observations in time.')
parser.add_argument('--nx_obs', type=int, default=30, help='Number of observations in space. Only used if obs_spacing=random.')
parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
parser.add_argument('--taylor_test', action='store_true', help='Run adjoint Taylor test and exit.')
parser.add_argument('--vtk', action='store_true', help='Write out timeseries to VTK file.')
parser.add_argument('--visualise', action='store_true', help='Visualise DAG.')
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

mesh = fd.PeriodicUnitIntervalMesh(args.nx, name="Domain mesh")
mesh_out = fd.UnitIntervalMesh(args.nx)

un, un1, stepper = burgers_stepper(nu, dt, theta, mesh)
V = un.function_space()

# true initial condition and perturbed starting guess
ic_target = noisy_double_sin(V, avg=args.ubar, mag=0.5, shift=-0.02, noise=None)
background = noisy_double_sin(V, avg=args.ubar, mag=0.5*args.prior_mag,
                              shift=args.prior_shift, noise=args.prior_noise, seed=args.seed)
background.topological.rename('Background')

uend_target = fd.Function(V, name="Target")

# target forward solution

Print("Running target forward model")

# observations taken on VOM
obs_points = [[x] for x in sorted(np.random.random_sample(args.nx_obs))]

obs_mesh = fd.VertexOnlyMesh(mesh, obs_points, name="Observation locations")
Vobs = fd.VectorFunctionSpace(obs_mesh, "DG", 0)


# observation operator
def H(x, name=None):
    hx = fd.assemble(interpolate(x, Vobs), ad_block_tag='Observation operator')
    if name is not None:
        hx.topological.rename(name)
    return hx


y = []
utargets = [ic_target.copy(deepcopy=True)]

# calculate "ground truth" observation values from target ic
t = 0.0
nsteps = int(0)
obs_times = [0]

un.assign(ic_target)
un1.assign(ic_target)
y.append(H(un, name=f'Observation {len(obs_times)-1}'))
while (t + 0.5*dt) <= args.tend:
    stepper.solve()
    un.assign(un1)
    t += dt
    nsteps += int(1)
    utargets.append(un.copy(deepcopy=True))
    if (nsteps % args.obs_freq) == 0:
        obs_times.append(nsteps)
        y.append(H(un, name=f'Observation {len(obs_times)-1}'))
Print(f"Number of timesteps {nsteps = }")
Print(f"Number of observations {len(y) = }")
Print(f"{obs_times = }")

# Initialise forward solution
Print("Setting up strong constraint 4DVar adjoint model")


# weighted l2 inner product
def wl2prod(x, w=1.0, ad_block_tag=None):
    return fd.assemble(fd.inner(x, w*x)*fd.dx, ad_block_tag=ad_block_tag)


observation_iprod = partial(wl2prod, w=R, ad_block_tag='Observation error')

# Initialise forward model from prior/background initial conditions
# and accumulate strong constraint functional as we go
ic_approx = background.copy(deepcopy=True, annotate=False)
ic_approx.topological.rename("Control")

continue_annotation()

un.assign(ic_approx)
un1.assign(ic_approx)

hx = []
observation_idx = 0
uapprox = [ic_approx.copy(deepcopy=True, annotate=False)]

# background error
background_err = ic_approx - background
J = wl2prod(background_err, B, ad_block_tag='Background error')

Print("Running forward model")

# initial observation error
hx.append(H(un, name=f'Model observation {observation_idx}'))
J += observation_iprod(hx[-1] - y[observation_idx])
observation_idx += 1

for i in range(nsteps):
    stepper.solve()
    un.assign(un1)
    uapprox.append(un.copy(deepcopy=True, annotate=False))

    if (i + 1) == obs_times[observation_idx]:
        hx.append(H(un, name=f'Model observation {observation_idx}'))
        J += observation_iprod(hx[-1] - y[observation_idx])

        observation_idx += 1
        if observation_idx == len(obs_times):
            break

Print("Setting up strong constraint 4DVar ReducedFunctional")
Jhat = ReducedFunctional(J, Control(ic_approx))
pause_annotation()
Jhat.optimize_tape()

if args.taylor_test:
    from firedrake.adjoint import taylor_to_dict
    from sys import exit
    Print("Running Taylor tests on strong constraint 4DVar reduced functional")
    h = fd.Function(V)
    for hdat in h.dat:
        hdat.data[:] = np.random.random_sample(hdat.data.shape)
    taylor_results = taylor_to_dict(Jhat, ic_approx, h)
    Print(f"{np.mean(taylor_results['R0']['Rate']) = }")
    Print(f"{np.mean(taylor_results['R1']['Rate']) = }")
    Print(f"{np.mean(taylor_results['R2']['Rate']) = }")
    exit()

Print("Minimizing strong constraint 4DVar functional")
options = {'disp': True, 'ftol': args.tol}
derivative_options = {'riesz_representation': 'l2'}
ic_opt = minimize(Jhat, options=options, method="L-BFGS-B",
                  derivative_options=derivative_options)

# calculate timeseries from optimised initial conditions
uopt = [ic_opt.copy(deepcopy=True)]
un.assign(ic_opt)
un1.assign(ic_opt)
for _ in range(nsteps):
    stepper.solve()
    un.assign(un1)
    uopt.append(un.copy(deepcopy=True))

Print(f"Initial functional: {Jhat(background)}")
Print(f"Final functional: {Jhat(ic_opt)}")
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
