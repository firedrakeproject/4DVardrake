# Burgers equation N-wave on a 1D periodic domain
import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from firedrake.adjoint import (continue_annotation, get_working_tape, pause_annotation,
                               Control, minimize)
from firedrake.adjoint.all_at_once_reduced_functional import AllAtOnceReducedFunctional
from burgers_utils import noisy_double_sin, burgers_stepper
import numpy as np
from functools import partial
import argparse
from sys import exit

np.set_printoptions(legacy='1.25')

Print = PETSc.Sys.Print

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
parser.add_argument('--n_obs', type=int, default=30, help='Number of observations in space. Only used if obs_spacing=random.')
parser.add_argument('--no_initial_obs', action='store_true', help='No observation at initial time.')
parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
parser.add_argument('--method', type=str, default='bfgs', help='Minimization method.')
parser.add_argument('--constraint', type=str, default='weak', choices=['weak', 'strong'], help='4DVar formulation to use.')
parser.add_argument('--taylor_test', action='store_true', help='Run adjoint Taylor test and exit.')
parser.add_argument('--vtk', action='store_true', help='Write out timeseries to VTK file.')
parser.add_argument('--vtk_file', type=str, default='burgers_4dvar', help='VTK file name.')
parser.add_argument('--progress', action='store_true', help='Show tape progress bar.')
parser.add_argument('--visualise', action='store_true', help='Visualise DAG.')
parser.add_argument('--dag_file', type=str, default=None, help='DAG file name.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

initial_observations = not args.no_initial_obs

Print("Setting up problem")
np.random.seed(args.seed)

# problem parameters
nu = 1/args.re
dt = args.cfl/(args.nx*args.ubar)
theta = args.theta

# error covariance inverse
B = fd.Constant(args.B)
R = fd.Constant(args.R)
Q = fd.Constant(args.Q)

mesh = fd.PeriodicUnitIntervalMesh(args.nx, name="Domain mesh")
mesh_out = fd.UnitIntervalMesh(args.nx)

un, un1, stepper = burgers_stepper(nu, dt, theta, mesh)
V = un.function_space()

# true initial condition and perturbed starting guess
ic_target = noisy_double_sin(V, avg=args.ubar, mag=0.5, shift=-0.02, noise=None)
background = noisy_double_sin(V, avg=args.ubar, mag=0.5*args.prior_mag,
                              shift=args.prior_shift, noise=args.prior_noise, seed=None)
background.topological.rename('Control 0')

uend_target = fd.Function(V, name="Target")

un.assign(ic_target)
un1.assign(ic_target)

# target forward solution

Print("Running target forward model")

# observations taken on VOM
if args.obs_spacing == 'equidistant':
    coords = mesh_out.coordinates.dat.data
    obs_points = [[coords[i]] for i in range(0, len(coords), args.obs_density)]
if args.obs_spacing == 'random':
    obs_points = [[x] for x in sorted(np.random.random_sample(args.n_obs))]

obs_mesh = fd.VertexOnlyMesh(mesh, obs_points, name="Observation locations")
Vobs = fd.VectorFunctionSpace(obs_mesh, "DG", 0)


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
obs_times = []
if initial_observations:
    obs_times.append(0)
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
Print("Setting up adjoint model")


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

tape = get_working_tape()
continue_annotation()

uapprox = [background.copy(deepcopy=True, annotate=False)]

Jhat = AllAtOnceReducedFunctional(
    Control(background),
    background_iprod=background_iprod,
    observation_iprod=observation_iprod if initial_observations else None,
    observation_err=partial(observation_err, 0, name='Model observation 0') if initial_observations else None,
    weak_constraint=(args.constraint == 'weak'))

un.assign(background)

Jhat.background.topological.rename("Background")

Print("Running forward model")

observation_idx = 1 if initial_observations else 0

for i in range(nsteps):
    un1.assign(un)
    stepper.solve()
    un.assign(un1)

    uapprox.append(un.copy(deepcopy=True, annotate=False))

    if (i + 1) == obs_times[observation_idx]:

        obs_error = partial(observation_err, observation_idx,
                            name=f'Model observation {observation_idx}')

        model_iprod = partial(wl2prod, w=Q, ad_block_tag=f'Model inner product {observation_idx}')

        Jhat.set_observation(un, obs_error,
                             observation_iprod=observation_iprod,
                             forward_model_iprod=model_iprod)

        observation_idx += 1
        if observation_idx >= len(obs_times):
            break

pause_annotation()

if args.taylor_test:
    from firedrake.adjoint import taylor_to_dict, taylor_test
    h = [fd.Function(V).zero() for _ in range(len(Jhat.controls))]
    for hi in h:
        hi.dat.data[:] = 0.1*np.random.random_sample(hi.dat.data.shape)
    ucs = [c.copy_data() for c in Jhat.controls]

    Print(f"{Jhat(ucs) = }")
    Print(f"{[fd.norm(d) for d in Jhat.derivative()] = }")
    Print("Updating ucs...")
    for ui, hi in zip(ucs, h):
        ui += hi
    Print("ucs updated")
    Print(f"{Jhat(ucs) = }")
    Print(f"{[fd.norm(d) for d in Jhat.derivative()] = }")

    Jhat.background_error.tape.visualise(output='dag_bkg_error.pdf')
    Jhat.background_rf.tape.visualise(output='dag_bkg_rf.pdf')

    for i in range(len(Jhat.observation_rfs)):
        Jhat.observation_rfs[i].tape.visualise(output=f'dag_obs_rf_{i}.pdf')
        Jhat.observation_errors[i].tape.visualise(output=f'dag_obs_err_{i}.pdf')

    for i in range(len(Jhat.forward_model_rfs)):
        Jhat.forward_model_rfs[i].tape.visualise(output=f'dag_model_rf_{i+1}.pdf')
        Jhat.forward_model_stages[i].tape.visualise(output=f'dag_model_stage_{i+1}.pdf')
        Jhat.forward_model_errors[i].tape.visualise(output=f'dag_model_error_{i+1}.pdf')

    Print("Running Taylor tests on AllAtOnceReducedFunctional")
    Print(f"{taylor_test(Jhat, ucs, h) = }")
    exit()

    taylor_results = taylor_to_dict(Jhat, ucs, h)
    Print(f"{np.mean(taylor_results['R0']['Rate']) = }")
    Print(f"{np.mean(taylor_results['R1']['Rate']) = }")
    Print(f"{np.mean(taylor_results['R2']['Rate']) = }")
    exit()

Print("Minimizing 4DVar functional")
if args.progress:
    tape.progress_bar = fd.ProgressBar

ucontrols = [c.control.copy(deepcopy=True) for c in Jhat.controls]

if args.method == 'bfgs':
    options = {'disp': True, 'maxcor': args.maxcor, 'ftol': args.ftol, 'gtol': args.gtol}
    uoptimised = minimize(Jhat, options=options, method="L-BFGS-B")
elif args.method == 'newton':
    options = {'disp': True, 'maxiter': args.maxcor, 'xtol': args.ftol}
    uoptimised = minimize(Jhat, options=options, method="Newton-CG")
else:
    raise ValueError("Unrecognised minimization method {args.method}")

Print(f"Initial functional: {Jhat(ucontrols)}")
Print(f"Final functional: {Jhat(uoptimised)}")

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

Print(f"Initial ic error: {fd.errornorm(background, ic_target)}")
Print(f"Final ic error: {fd.errornorm(uopt0, ic_target)}")
Print(f"Initial terminal error: {fd.errornorm(uapprox[-1], utargets[-1])}")
Print(f"Final terminal error: {fd.errornorm(uopt[-1], utargets[-1])}")
# if Jhat.weak_constraint:
#     Print(f"Final terminal error: {fd.errornorm(uopt_weak[-1], utargets[-1])}")

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
