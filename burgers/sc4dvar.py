# Burgers equation N-wave on a 1D periodic domain
import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.output import VTKFile
from firedrake.__future__ import Interpolator, interpolate
from firedrake.adjoint import (continue_annotation, get_working_tape, pause_annotation,
                               Control, ReducedFunctional, minimize)
import numpy as np
from sys import exit
import argparse

Print = PETSc.Sys.Print


class InterpWriter:
    def __init__(self, fname, Vsrc, Vout, vnames=None):
        vnames = [] if vnames is None else vnames
        self.Vsrc, self.Vout = Vsrc, Vout
        self.ofile = VTKFile(fname)
        self.usrc = fd.Function(Vsrc)
        if vnames is None:
            self.uouts = [fd.Function(Vout)]
        else:
            self.uouts = [fd.Function(Vout, name=name)
                          for name in vnames]
        self.interpolator = Interpolator(self.usrc, Vout)

    def write(self, *args, t=None):
        for us, uo in zip(args, self.uouts):
            self.usrc.assign(us)
            uo.assign(fd.assemble(self.interpolator.interpolate()))
        self.ofile.write(*self.uouts, time=t)


def initial(V, avg=1, mag=0.5, shift=0, noise=None, seed=None):
    x, = fd.SpatialCoordinate(V.mesh())
    ic = fd.project(fd.as_vector([avg + mag*fd.sin(2*fd.pi*(x+shift))
                                  + 0.2*mag*fd.cos(6*fd.pi*(x-shift))]), V)
    if seed is not None:
        np.random.seed(seed)
    if noise is not None:
        for dat in ic.dat:
            dat.data[:] += noise*np.random.random_sample(dat.data.shape)
    return ic


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
parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance of optimiser.')
parser.add_argument('--prior_mag', type=float, default=1.1, help='Magnitude of background vs truth.')
parser.add_argument('--prior_shift', type=float, default=0.05, help='Phase shift in background vs truth.')
parser.add_argument('--prior_noise', type=float, default=0.05, help='Noise magnitude in background.')
parser.add_argument('--B', type=float, default=1e-1, help='Background trust weighting.')
parser.add_argument('--R', type=float, default=1, help='Observation trust weighting.')
parser.add_argument('--Rfinal', type=float, default=0, help='Terminal cost trust weighting.')
parser.add_argument('--obs_spacing', type=str, default='random', choices=['random', 'equidistant'], help='How observation points are distributed in space.')
parser.add_argument('--obs_freq', type=int, default=10, help='Frequency of observations in time.')
parser.add_argument('--obs_density', type=int, default=10, help='Frequency of observations in space. Only used if obs_spacing=equidistant.')
parser.add_argument('--n_obs', type=int, default=10, help='Number of observations in space. Only used if obs_spacing=random.')
parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
parser.add_argument('--taylor_test', action='store_true', help='Run adjoint Taylor test.')
parser.add_argument('--progress', action='store_true', help='Show tape progress bar.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

Print("Setting up problem")
np.random.seed(args.seed)

# problem parameters
nu = fd.Constant(1/args.re)
dt = args.cfl/(args.nx*args.ubar)
dt1 = fd.Constant(1/dt)
theta = fd.Constant(args.theta)

mesh = fd.PeriodicUnitIntervalMesh(args.nx)
mesh_out = fd.UnitIntervalMesh(args.nx)

V = fd.VectorFunctionSpace(mesh, "CG", 2)

un = fd.Function(V, name="Velocity")
un1 = fd.Function(V, name="VelocityNext")

# true initial condition and perturbed starting guess
ic_target = initial(V, avg=args.ubar, mag=0.5, shift=-0.02, noise=None)
background = initial(V, avg=args.ubar, mag=0.5*args.prior_mag,
                     shift=args.prior_shift, noise=args.prior_noise)

uend_target = fd.Function(V, name="Target")

un.assign(ic_target)
un1.assign(ic_target)


def mass(u, v):
    return fd.inner(u, v)*fd.dx


def tendency(u, v):
    A = fd.inner(fd.dot(u, fd.nabla_grad(u)), v)
    D = nu*fd.inner(fd.grad(u), fd.grad(v))
    return (A + D)*fd.dx


v = fd.TestFunction(V)
M = dt1*mass(un1-un, v)
A = theta*tendency(un1, v) + (1-theta)*tendency(un, v)

F = M + A

solver = fd.NonlinearVariationalSolver(
    fd.NonlinearVariationalProblem(F, un1))

# output on non-periodic mesh
V_out = fd.VectorFunctionSpace(mesh_out, "CG", 1)

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
y.append(H(un))
while (t <= args.tend):
    solver.solve()
    un.assign(un1)
    t += dt
    utargets.append(un.copy(deepcopy=True))
    if ((nsteps+1) % args.obs_freq) == 0:
        y.append(H(un))
    nsteps += int(1)
uend_target.assign(un)
Print(f"Number of timesteps {nsteps = }")

# Initialise forward solution
Print("Setting up adjoint model")

# Initialise forward model from prior/background initial conditions
ic_approx = background.copy(deepcopy=True)
continue_annotation()
un.assign(ic_approx)
un1.assign(ic_approx)

hx = []
uapprox = [ic_approx.copy(deepcopy=True)]

Print("Running forward model")
tape = get_working_tape()
hx.append(H(un))
for i in range(nsteps):
    solver.solve()
    un.assign(un1)
    uapprox.append(un.copy(deepcopy=True))
    if ((i+1) % args.obs_freq) == 0:
        hx.append(H(un))
uend_approx = un.copy(deepcopy=True)

Print("Setting up ReducedFunctional")
B = fd.Constant(args.B)
R = fd.Constant(args.R)
Rf = fd.Constant(args.Rfinal)

# How far from final solution?
terminal_err = uend_approx - uend_target
# How far from prior ic?
bkg_err = ic_approx - background
# How far from observations?
obs_err = sum(fd.inner(hi-yi, hi-yi)*fd.dx for hi, yi in zip(hx, y))

J = fd.assemble(Rf*fd.inner(terminal_err, terminal_err)*fd.dx)
J += fd.assemble(B*fd.inner(bkg_err, bkg_err)*fd.dx)
J += fd.assemble(R*obs_err)

ic = Control(ic_approx)
Jhat = ReducedFunctional(J, ic)

if args.taylor_test:
    from firedrake.adjoint import taylor_test
    h = fd.Function(V)
    h.dat.data[:] = np.random.random_sample(h.dat.data.shape)
    Print(f"{taylor_test(Jhat, ic_approx, h) = }")
    exit()

Print("Minimizing 4DVar functional")
if args.progress:
    tape.progress_bar = fd.ProgressBar
ic_opt = minimize(Jhat, options={'disp': True}, method="L-BFGS-B", tol=args.tol)

Print(f"Initial functional: {Jhat(background)}")
Print(f"Final functional: {Jhat(ic_opt)}")

tape.clear_tape()
pause_annotation()

uopt = [ic_opt.copy(deepcopy=True)]

# calculate timeseries from optimised initial conditions
un.assign(ic_opt)
un1.assign(ic_opt)
for _ in range(nsteps):
    solver.solve()
    un.assign(un1)
    uopt.append(un.copy(deepcopy=True))
uend_opt = un.copy(deepcopy=True)

Print(f"Initial ic error: {fd.errornorm(background, ic_target) = }")
Print(f"Final ic error: {fd.errornorm(ic_opt, ic_target) = }")
Print(f"Initial terminal error: {fd.errornorm(uend_approx, uend_target) = }")
Print(f"Final terminal error: {fd.errornorm(uend_opt, uend_target) = }")

vnames = ["TargetVelocity", "InitialGuess", "OptimisedVelocity"]
write = InterpWriter("output/burgers_target.pvd", V, V_out, vnames).write
for i, us in enumerate(zip(utargets, uapprox, uopt)):
    write(*us, t=i*dt)

tape.clear_tape()
