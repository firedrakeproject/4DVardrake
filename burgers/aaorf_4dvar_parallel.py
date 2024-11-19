# Burgers equation N-wave on a 1D periodic domain
import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from firedrake.adjoint import (continue_annotation, pause_annotation, stop_annotating,
                               get_working_tape, Control, minimize)
from firedrake.adjoint.all_at_once_reduced_functional import AllAtOnceReducedFunctional
from burgers_utils import noisy_double_sin, burgers_stepper
import numpy as np
from functools import partial
import argparse
from sys import exit
from math import ceil


def scalarSend(comm, x, dtype=float, **kwargs):
    comm.Send(x*np.ones(1, dtype=dtype), **kwargs)


def scalarRecv(comm, dtype=float, **kwargs):
    xtmp = np.zeros(1, dtype=dtype)
    comm.Recv(xtmp, **kwargs)
    return xtmp[0]


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
parser.add_argument('--prior_mag', type=float, default=1.1, help='Magnitude of background vs truth.')
parser.add_argument('--prior_shift', type=float, default=0.05, help='Phase shift in background vs truth.')
parser.add_argument('--prior_noise', type=float, default=0.05, help='Noise magnitude in background.')
parser.add_argument('--B', type=float, default=1e-1, help='Background trust weighting.')
parser.add_argument('--Q', type=float, default=1.0, help='Model trust weighting.')
parser.add_argument('--R', type=float, default=1.0, help='Observation trust weighting.')
parser.add_argument('--obs_spacing', type=str, default='random', choices=['random', 'equidistant'], help='How observation points are distributed in space.')
parser.add_argument('--obs_freq', type=int, default=10, help='Frequency of observations in time.')
parser.add_argument('--obs_density', type=int, default=30, help='Frequency of observations in space. Only used if obs_spacing=equidistant.')
parser.add_argument('--nx_obs', type=int, default=30, help='Number of observations in space. Only used if obs_spacing=random.')
parser.add_argument('--no_initial_obs', action='store_true', help='No observation at initial time.')
parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
parser.add_argument('--method', type=str, default='bfgs', help='Minimization method.')
parser.add_argument('--constraint', type=str, default='weak', choices=['weak', 'strong'], help='4DVar formulation to use.')
parser.add_argument('--nchunks', type=int, default=1, help='Number of chunks in time.')
parser.add_argument('--taylor_test', action='store_true', help='Run adjoint Taylor test and exit.')
parser.add_argument('--single_evaluation', action='store_true', help='Evaluate the functional and gradient once and exit.')
parser.add_argument('--vtk', action='store_true', help='Write out timeseries to VTK file.')
parser.add_argument('--vtk_file', type=str, default='burgers_4dvar', help='VTK file name.')
parser.add_argument('--progress', action='store_true', help='Show tape progress bar.')
parser.add_argument('--visualise', action='store_true', help='Visualise DAG.')
parser.add_argument('--dag_file', type=str, default=None, help='DAG file name.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--verbose', action='store_true', help='Print a load of stuff.')

args = parser.parse_known_args()
args = args[0]

Print = lambda *ags, **kws: PETSc.Sys.Print(*ags, **kws) if args.verbose else None

if args.show_args:
    PETSc.Sys.Print(args)

##################################################
### Process script arguments
##################################################

initial_observations = not args.no_initial_obs

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
nranks_space = global_comm.size // args.nchunks
ensemble = fd.Ensemble(global_comm, nranks_space)
last_rank = ensemble.ensemble_comm.size - 1
trank = ensemble.ensemble_comm.rank

##################################################
### Build the mesh and timestepper
##################################################

mesh = fd.PeriodicUnitIntervalMesh(args.nx, name="Domain mesh", comm=ensemble.comm)
mesh_out = fd.UnitIntervalMesh(args.nx, comm=ensemble.comm)

un, un1, stepper = burgers_stepper(nu, dt, theta, mesh)
V = un.function_space()

# true initial condition and perturbed starting guess
ic_target = noisy_double_sin(V, avg=args.ubar, mag=0.5, shift=-0.02, noise=None)
background = noisy_double_sin(V, avg=args.ubar, mag=0.5*args.prior_mag,
                                  shift=args.prior_shift, noise=args.prior_noise, seed=None)
background.topological.rename('Control 0')

# target forward solution

global_comm.Barrier()
PETSc.Sys.Print("Running target forward model")
global_comm.Barrier()

##################################################
### Select observation locations
##################################################

# observations taken on VOM
if args.obs_spacing == 'equidistant':
    coords = mesh_out.coordinates.dat.data
    obs_points = [[coords[i]] for i in range(0, len(coords), args.obs_density)]
if args.obs_spacing == 'random':
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

# calculate "ground truth" observation values from target ic
if trank == 0:
    utargets = [ic_target.copy(deepcopy=True)]
    t = 0.0
    nsteps = int(0)
    un.assign(ic_target)
    un1.assign(ic_target)
    # Print(f"Time = {str(round(t, 4)).ljust(6)} | {fd.norm(un)    = }", comm=ensemble.comm)
    if initial_observations:
        obs_times.append(0)
        yn = H(un, name=f'Observation {len(obs_times)-1}')
        y.append(yn)
        # Print(f"              | {fd.norm(yn)    = }", comm=ensemble.comm)

else:
    ensemble.recv(un, source=trank-1, tag=trank+000)
    t = scalarRecv(ensemble.ensemble_comm, dtype=float, source=trank-1, tag=trank+100)
    nsteps = scalarRecv(ensemble.ensemble_comm, dtype=int, source=trank-1, tag=trank+200)


for k in range(nlocal_observations):
    for i in range(nt):
        un1.assign(un)
        stepper.solve()
        un.assign(un1)
        t += dt
        nsteps += 1
        utargets.append(un.copy(deepcopy=True))

    assert (nsteps % args.obs_freq) == 0
    obs_times.append(nsteps)
    yn = H(un, name=f'Observation {len(obs_times)-1}')
    y.append(yn)
Print(f"{trank = } | {obs_times = }", comm=ensemble.comm)

if trank != last_rank:
    ensemble.send(un, dest=trank+1, tag=trank+1+000)
    scalarSend(ensemble.ensemble_comm, t, dtype=float,  dest=trank+1, tag=trank+1+100)
    scalarSend(ensemble.ensemble_comm, nsteps, dtype=int,  dest=trank+1, tag=trank+1+200)

if trank == last_rank:
    PETSc.Sys.Print(f"Number of timesteps {nsteps = }", comm=ensemble.comm)

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

# Only initial rank needs data for initial conditions or time
if trank == 0:
    background_iprod0 = background_iprod
    if initial_observations:
        observation_iprod0 = observation_iprod
        observation_err0 = partial(observation_err, 0, name='Model observation 0')
    else:
        observation_iprod0 = None
        observation_err0 = None
else:
    background_iprod0 = None
    observation_iprod0 = None
    observation_err0 = None

##################################################
### Create the 4dvar reduced functional
##################################################

# first rank has one extra control for the initial conditions
nlocal_controls = nlocal_observations + (1 if trank == 0 else 0)
aaofunc = fd.EnsembleFunction(ensemble, [V for _ in range(nlocal_controls)])

# initial guess at first control (initial condition) is the background prior
aaofunc.subfunctions[0].assign(background)

continue_annotation()

Jhat = AllAtOnceReducedFunctional(
    Control(aaofunc),
    background=background,
    background_iprod=background_iprod0,
    observation_iprod=observation_iprod0,
    observation_err=observation_err0,
    weak_constraint=(args.constraint == 'weak'))

Jhat.background.topological.rename("Background")

global_comm.Barrier()
PETSc.Sys.Print("Running forward model")
global_comm.Barrier()

observation_idx = 1 if initial_observations and trank == 0 else 0
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
        local_obs_idx = ctx.local_index + obs_offset

        # index of this observation globally
        global_obs_idx = ctx.global_index + (1 if initial_observations else 0)

        obs_error = partial(observation_err, local_obs_idx,
                            name=f'Model observation {global_obs_idx}')

        model_iprod = partial(wl2prod, w=Q,
                              ad_block_tag=f'Model inner product {global_obs_idx}')

        # record the observation at the end of the stage
        stage.set_observation(un, obs_error,
                              observation_iprod=observation_iprod,
                              forward_model_iprod=model_iprod)

global_comm.Barrier()

pause_annotation()

global_comm.Barrier()

##################################################
### Check the AllAtOnceReducedFunctional
##################################################

global_comm.Barrier()
ucontrol = Jhat.control.copy_data()
global_comm.Barrier()

if args.single_evaluation:
    PETSc.Sys.Print("Evaluate Jhat in parallel")
    global_comm.Barrier()

    for i, uc in enumerate(ucontrol.subfunctions):
        # index of first control on chunk
        obs_per_chunk = nlocal_observations
        offset = 0 if trank == 0 else 1 + trank*obs_per_chunk
        scale = 1 + 0.1*(1 + offset+i)
        uc *= scale
        PETSc.Sys.Print(f"{trank = } | {i = } | {scale = } | {fd.norm(uc) = }", comm=ensemble.comm)
    global_comm.Barrier()
    Print(f"{trank = } | {Jhat(ucontrol) = }", comm=ensemble.comm)
    global_comm.Barrier()

    global_comm.Barrier()
    PETSc.Sys.Print("Evaluate Jhat.derivative in parallel")
    global_comm.Barrier()
    deriv = Jhat.derivative()
    for i, d in enumerate(deriv.subfunctions):
        Print(f"{trank = } | {i = } | {fd.norm(d) = }", comm=ensemble.comm)

    exit()

if args.taylor_test:
    from firedrake.adjoint import taylor_to_dict, taylor_test

    PETSc.Sys.Print("Evaluate Jhat and derivative")

    h = 0.1*ucontrol.copy()

    global_comm.Barrier()
    Print(f"{trank= } | {Jhat(ucontrol) = }", comm=ensemble.comm)
    Print(f"{trank = } | {[fd.norm(d) for d in Jhat.derivative().subfunctions] = }", comm=ensemble.comm)
    global_comm.Barrier()
    Print("Updating ucontrol...")
    for ui, hi in zip(ucontrol.subfunctions, h.subfunctions):
        ui += hi
    global_comm.Barrier()
    Print("ucontrol updated")
    global_comm.Barrier()
    PETSc.Sys.Print("Evaluate Jhat and derivative after perturbation")
    Print(f"{Jhat(ucontrol) = }", comm=ensemble.comm)
    global_comm.Barrier()
    Print(f"{trank = } | {[fd.norm(d) for d in Jhat.derivative().subfunctions] = }", comm=ensemble.comm)
    global_comm.Barrier()

    PETSc.Sys.Print("Running Taylor tests on AllAtOnceReducedFunctional")
    global_comm.Barrier()
    PETSc.Sys.Print(f"{trank = } | {taylor_test(Jhat, ucontrol, h) = }")
    exit()


##################################################
### Minimize
##################################################

tape = get_working_tape()

PETSc.Sys.Print("Minimizing 4DVar functional")
if args.progress:
    tape.progress_bar = fd.ProgressBar

ucontrols = [c.control.copy(deepcopy=True) for c in Jhat.controls]

if args.method == 'bfgs':
    options = {'disp': trank == 0, 'maxcor': args.maxcor, 'ftol': args.ftol, 'gtol': args.gtol}
    uoptimised = minimize(Jhat, options=options, method="L-BFGS-B")
elif args.method == 'newton':
    options = {'disp': trank == 0, 'maxiter': args.maxcor, 'xtol': args.ftol}
    uoptimised = minimize(Jhat, options=options, method="Newton-CG")
else:
    raise ValueError("Unrecognised minimization method {args.method}")

print(f"{trank = } | {len(ucontrols) = }")
print(f"{trank = } | {len(uoptimised) = }")

for uc in Jhat.controls:
    uc = uc.control
    print(f"{trank = } | {uc = }")

for uo in uoptimised:
    print(f"{trank = } | {uo = } | {uo.ufl_operands[0] = }")

PETSc.Sys.Print(f"Initial functional: {Jhat(ucontrols)}")
PETSc.Sys.Print(f"Final functional: {Jhat(uoptimised)}")

if args.visualise:
    Jhat.optimize_tape()
    if args.dag_file is None:
        filename = f'dag_{args.constraint}_aaorf'
    else:
        filename = args.dag_file
    tape.visualise(output=f'{filename}.pdf')
tape.clear_tape()

uopt0 = uoptimised[0] if Jhat.weak_constraint else uoptimised
uopt = [uopt0.copy(deepcopy=True)]

# calculate timeseries from optimised initial conditions
un.assign(uoptimised[0])
un1.assign(uoptimised[0])
for _ in range(nsteps):
    stepper.solve()
    un.assign(un1)
    uopt.append(un.copy(deepcopy=True))

if Jhat.weak_constraint:
    uopt_weak = [uoptimised[0].copy(deepcopy=True)]
    un.assign(uopt_weak[0])
    un1.assign(uopt_weak[0])
    observation_idx = 1 if initial_observations else 0
    for i in range(nsteps):
        un1.assign(un)
        stepper.solve()
        un.assign(un1)
        if (i+1) == obs_times[observation_idx]:
            un.assign(uoptimised[observation_idx])
            observation_idx += 1
        uopt_weak.append(un.copy(deepcopy=True))
        if observation_idx >= len(obs_times):
            break

PETSc.Sys.Print(f"Initial ic error: {fd.errornorm(background, ic_target)}")
PETSc.Sys.Print(f"Final ic error: {fd.errornorm(uopt0, ic_target)}")
PETSc.Sys.Print(f"Initial terminal error: {fd.errornorm(uapprox[-1], utargets[-1])}")
PETSc.Sys.Print(f"Final terminal error: {fd.errornorm(uopt[-1], utargets[-1])}")
if Jhat.weak_constraint:
    PETSc.Sys.Print(f"Final terminal error: {fd.errornorm(uopt_weak[-1], utargets[-1])}")

if args.vtk:
    from burgers_utils import InterpWriter
    # output on non-periodic mesh
    V_out = fd.VectorFunctionSpace(mesh_out, "CG", 1)
    vnames = ["TargetVelocity", "InitialGuess", "OptimisedVelocity", "OptimisedInitial"]
    if Jhat.weak_constraint:
        vfuncs = (utargets, uapprox, uopt_weak, uopt)
    else:
        vfuncs = (utargets, uapprox, uopt, uopt)
    write = InterpWriter(f"output/{args.vtk_file}.pvd", V, V_out, vnames).write
    for i, us in enumerate(zip(*vfuncs)):
        write(*us, t=i*dt)

tape.clear_tape()
