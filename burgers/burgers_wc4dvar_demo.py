# Burgers equation N-wave on a 1D periodic domain
import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from firedrake.adjoint import (continue_annotation, pause_annotation,
                               get_working_tape, Control, minimize)
from firedrake.adjoint import FourDVarReducedFunctional
from burgers_utils import noisy_double_sin, burgers_stepper
import numpy as np
from functools import partial
import argparse
from sys import exit
from math import ceil

np.set_printoptions(legacy='1.25')

parser = argparse.ArgumentParser(
    description='Weak constraint 4DVar for the viscous Burgers equation.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=100, help='Number of elements.')
parser.add_argument('--cfl', type=float, default=1.0, help='Approximate Courant number.')
parser.add_argument('--ubar', type=float, default=1.0, help='Average initial velocity.')
parser.add_argument('--tend', type=float, default=1.0, help='Final integration time.')
parser.add_argument('--re', type=float, default=1e2, help='Approximate Reynolds number.')
parser.add_argument('--theta', type=float, default=0.5, help='Implicit timestepping parameter.')
parser.add_argument('--ftol', type=float, default=1e-2, help='Optimiser tolerance for relative function reduction.')
parser.add_argument('--gtol', type=float, default=1e-2, help='Optimiser tolerance for gradient norm.')
parser.add_argument('--maxcor', type=int, default=20, help='Optimiser max corrections.')
parser.add_argument('--prior_mag', type=float, default=1.05, help='Magnitude of background vs truth.')
parser.add_argument('--prior_shift', type=float, default=0.025, help='Phase shift in background vs truth.')
parser.add_argument('--prior_noise', type=float, default=0.025, help='Noise magnitude in background.')
parser.add_argument('--B', type=float, default=1e-1, help='Background trust weighting.')
parser.add_argument('--Q', type=float, default=1e0, help='Model trust weighting.')
parser.add_argument('--R', type=float, default=1e1, help='Observation trust weighting.')
parser.add_argument('--obs_freq', type=int, default=10, help='Frequency of observations in time.')
parser.add_argument('--nx_obs', type=int, default=30, help='Number of observations in space. Only used if obs_spacing=random.')
parser.add_argument('--no_initial_obs', action='store_true', help='No observation at initial time.')
parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
parser.add_argument('--method', type=str, default='bfgs', help='Minimization method.')
parser.add_argument('--constraint', type=str, default='weak', choices=['weak', 'strong'], help='4DVar formulation to use.')
parser.add_argument('--nchunks', type=int, default=1, help='Number of chunks in time.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

##################################################
### Process script arguments
##################################################

PETSc.Sys.Print("Setting up problem")
np.random.seed(args.seed)

# problem parameters
nu = 1/args.re
dt = args.cfl/(args.nx*args.ubar)
theta = args.theta

# error covariance inverse
B = fd.Constant(args.B)
R = fd.Constant(args.R)
Q = fd.Constant(args.Q)

# do the chunks correspond to the observation times?
Nt = ceil(args.tend/dt)
nglobal_observations = Nt//args.obs_freq
assert (nglobal_observations % args.nchunks) == 0
nlocal_observations = nglobal_observations//args.nchunks
nt = Nt//nglobal_observations

PETSc.Sys.Print(f"Total timesteps = {Nt}")
PETSc.Sys.Print(f"Local timesteps = {nt}")
PETSc.Sys.Print(f"Total observations = {nglobal_observations}")
PETSc.Sys.Print(f"Local observations = {nlocal_observations}")

global_comm = fd.COMM_WORLD
if global_comm.size % args.nchunks != 0:
    raise ValueError("Number of time-chunks must exactly divide size of COMM_WORLD")
ensemble = fd.Ensemble(global_comm, global_comm.size//args.nchunks)
last_rank = ensemble.ensemble_comm.size - 1
trank = ensemble.ensemble_comm.rank

initial_observations = not args.no_initial_obs if trank == 0 else False

##################################################
### Build the mesh and timestepper
##################################################

mesh = fd.PeriodicUnitIntervalMesh(args.nx, name="Domain mesh", comm=ensemble.comm)
mesh_out = fd.UnitIntervalMesh(args.nx, comm=ensemble.comm)

un, un1, stepper = burgers_stepper(nu, dt, theta, mesh)
V = un.function_space()

##################################################
### Select observation locations
##################################################

# observations taken on VOM
obs_points = [[x] for x in sorted(np.random.random_sample(args.nx_obs))]

obs_mesh = fd.VertexOnlyMesh(mesh, obs_points, name="Observation locations")
Vobs = fd.VectorFunctionSpace(obs_mesh, "DG", 0)


def H(x, name=None):
    hx = fd.assemble(interpolate(x, Vobs), ad_block_tag='Observation operator')
    if name is not None:
        hx.topological.rename(name)
    return hx


y = []
utargets = []
obs_times = []

##################################################
### Calculate "ground truth" data
##################################################

# true initial condition and perturbed starting guess
ic_target = noisy_double_sin(V, avg=args.ubar, mag=0.5, shift=-0.02, noise=None)
background = noisy_double_sin(V, avg=args.ubar, mag=0.5*args.prior_mag,
                              shift=args.prior_shift, noise=args.prior_noise, seed=None)
background.rename('Control 0')

# target forward solution

global_comm.Barrier()
PETSc.Sys.Print("Running target forward model")
global_comm.Barrier()

t = 0.0
nsteps = 0

if trank == 0 and initial_observations:
    un.assign(ic_target)
    obs_times.append(0)
    y.append(H(un, name=f'Observation {len(obs_times)-1}'))
    utargets.append(un.copy(deepcopy=True))

with ensemble.sequential(t=t, nsteps=nsteps, un=un) as ctx:
    un.assign(ctx.un)
    for k in range(nlocal_observations):
        for i in range(nt):
            un1.assign(un)
            stepper.solve()
            un.assign(un1)

            ctx.t += dt
            ctx.nsteps += 1

        utargets.append(un.copy(deepcopy=True))
        obs_times.append(nsteps)

        y.append(H(un, name=f'Observation {len(obs_times)-1}'))

    nsteps = ctx.nsteps

# Initialise forward solution
global_comm.Barrier()
PETSc.Sys.Print("Setting up adjoint model")
global_comm.Barrier()


# weighted l2 inner product
def wl2prod(x, w=1.0, ad_block_tag=None):
    return fd.assemble(fd.inner(x, w*x)*fd.dx, ad_block_tag=ad_block_tag)


def observation_err(i, state, name=None):
    return fd.Function(Vobs, name=f'Observation error H{i}(x{i}) - y{i}').assign(H(state, name) - y[i], ad_block_tag=f"Observation error calculation {i}")


background_iprod = partial(wl2prod, w=B, ad_block_tag='Background inner product')
observation_iprod = partial(wl2prod, w=R, ad_block_tag='Observation inner product')
model_iprod = partial(wl2prod, w=Q, ad_block_tag='Model inner product')

# Initialise forward model from prior/background initial conditions
# and accumulate weak constraint functional as we go

uapprox = [background.copy(deepcopy=True, annotate=False)]

##################################################
### Create the 4dvar reduced functional
##################################################

background_iprod0 = background_iprod
if initial_observations:
    observation_iprod0 = observation_iprod
    observation_err0 = partial(observation_err, 0, name='Model observation 0')
else:
    observation_iprod0 = None
    observation_err0 = None

## Make sure this is the only point we are requiring user to know partition specifics

# first rank has one extra control for the initial conditions
nlocal_controls = nlocal_observations + (1 if trank == 0 else 0)

aaofunc = fd.EnsembleFunction(ensemble, [V for _ in range(nlocal_controls)])

if trank == 0:
    aaofunc.subfunctions[0].assign(background)

continue_annotation()

Jhat = FourDVarReducedFunctional(
    Control(aaofunc),
    background_iprod=background_iprod0,
    observation_iprod=observation_iprod0,
    observation_err=observation_err0,
    weak_constraint=(args.constraint == 'weak'))

Jhat.background.topological.rename("Background")

global_comm.Barrier()
PETSc.Sys.Print("Running forward model")
global_comm.Barrier()

observation_idx = 1 if initial_observations else 0
obs_offset = observation_idx

##################################################
### Record the forward model and observations
##################################################

# t and nsteps will be passed from one stage to another (including between ensemble members)
with Jhat.recording_stages(t=0.0, nsteps=0) as stages:

    for stage, ctx in stages:

        # start forward model for this stage
        un.assign(stage.control)

        for i in range(nt):
            un1.assign(un)
            stepper.solve()
            un.assign(un1)

            # increment the time and timestep
            ctx.t += dt
            ctx.nsteps += 1

            # stash the timeseries for plotting
        uapprox.append(un.copy(deepcopy=True, annotate=False))

        # index of the observation data for this stage on this ensemble member
        local_obs_idx = stage.local_index + obs_offset

        obs_error = partial(
            observation_err, local_obs_idx,
            name=f'Model observation {stage.observation_index}')

        model_iprod = partial(wl2prod, w=Q,
                              ad_block_tag=f'Model inner product {stage.observation_index}')

        # record the observation at the end of the stage
        stage.set_observation(un, obs_error,
                              observation_iprod=observation_iprod,
                              forward_model_iprod=model_iprod)

global_comm.Barrier()

pause_annotation()

global_comm.Barrier()


##################################################
### Minimize
##################################################

PETSc.Sys.Print("Minimizing 4DVar functional")

ucontrol = Jhat.control.copy()

# minimiser should be given the derivative not the gradient
derivative_options = {'riesz_representation': 'l2'}

options = {
    'disp': trank == 0,
    'maxcor': args.maxcor,
    'ftol': args.ftol,
    'gtol': args.gtol
}

uoptimised = minimize(Jhat, options=options, method="L-BFGS-B",
                       derivative_options=derivative_options)

uopts = uoptimised.subfunctions

global_comm.Barrier()
PETSc.Sys.Print(f"Initial functional: {Jhat(ucontrol)}")
PETSc.Sys.Print(f"Final functional: {Jhat(uoptimised)}")
global_comm.Barrier()
if trank == 0:
    PETSc.Sys.Print(f"Initial ic error: {fd.errornorm(background, ic_target)}", comm=ensemble.comm)
    PETSc.Sys.Print(f"Final ic error: {fd.errornorm(uopts[0], ic_target)}", comm=ensemble.comm)
global_comm.Barrier()
if trank == last_rank:
    PETSc.Sys.Print(f"Initial terminal error: {fd.errornorm(uapprox[-1], utargets[-1])}", comm=ensemble.comm)
    PETSc.Sys.Print(f"Final terminal error: {fd.errornorm(uopts[-1], utargets[-1])}", comm=ensemble.comm)
