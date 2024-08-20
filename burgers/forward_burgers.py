import firedrake as fd
from firedrake.petsc import PETSc
import argparse
from burgers_utils import InterpWriter, burgers_stepper, noisy_double_sin

Print = PETSc.Sys.Print

parser = argparse.ArgumentParser(
    description='Timestepping of 1D viscous Burgers equation using trapezium rule.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=100, help='Number of elements.')
parser.add_argument('--cfl', type=float, default=1.0, help='Approximate Courant number.')
parser.add_argument('--ubar', type=float, default=1.0, help='Average initial velocity.')
parser.add_argument('--tend', type=float, default=1.0, help='Final integration time.')
parser.add_argument('--re', type=float, default=1e2, help='Approximate Reynolds number.')
parser.add_argument('--theta', type=float, default=0.5, help='Implicit timestepping parameter.')
parser.add_argument('--phase_shift', type=float, default=-0.02, help='Phase shift of initial condition.')
parser.add_argument('--noise', type=float, default=0.00, help='Phase shift of initial condition.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()[0]

if args.show_args:
    Print(args)

# problem parameters
nu = args.ubar/args.re
dt = args.cfl/(args.nx*args.ubar)
theta = args.theta

mesh = fd.PeriodicUnitIntervalMesh(args.nx)

un, un1, stepper = burgers_stepper(nu, dt, theta, mesh)
V = un.function_space()

x, = fd.SpatialCoordinate(mesh)
ic = noisy_double_sin(V, avg=args.ubar, mag=0.5,
                      shift=args.phase_shift,
                      noise=args.noise)

un.assign(ic)
un1.assign(ic)

# output to non-periodic mesh
V_out = fd.VectorFunctionSpace(fd.UnitIntervalMesh(args.nx), "CG", 1)
write = InterpWriter("output/burgers.pvd", V, V_out, "Velocity").write
write(un, t=0)

t = 0.0
while (t <= args.tend):
    stepper.solve()
    un.assign(un1)
    t += dt
    write(un, t=t)
