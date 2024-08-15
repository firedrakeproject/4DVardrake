# Burgers equation N-wave on a 1D periodic domain
import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.output import VTKFile
from firedrake.__future__ import Interpolator
import numpy as np
import argparse

Print = PETSc.Sys.Print


# n-wave solution with analytic non-periodic solution
def nwave(x, t, nu):
    exp = np.exp if isinstance(x, np.ndarray) else fd.exp
    pi = np.pi if isinstance(x, np.ndarray) else fd.pi
    fnt1 = 4*nu*(t+1)
    etp2x = -8*t + 2*x
    m4tpx = -4*t + x
    return (-2*nu*(-etp2x*exp(-m4tpx**2/fnt1)/fnt1
                   - (etp2x - 4*pi)*exp(-(m4tpx - 2*pi)**2/fnt1)/fnt1)
            / (exp(-(m4tpx - 2*pi)**2/fnt1) + exp(-m4tpx**2/fnt1)) + 4)


# interpolate solution to non-periodic mesh to write output
class InterpWriter:
    def __init__(self, fname, Vsrc, Vout, vname=None):
        self.Vsrc, self.Vout = Vsrc, Vout
        self.ofile = VTKFile(fname)
        self.usrc = fd.Function(Vsrc)
        self.uout = fd.Function(Vout, name=vname)
        self.interpolator = Interpolator(self.usrc, Vout)

    def write(self, u, t=None):
        self.usrc.assign(u)
        self.uout.assign(fd.assemble(self.interpolator.interpolate()))
        self.ofile.write(self.uout, time=t)


def initial(V, avg=1, mag=0.5, shift=0, noise=None, seed=123):
    ic = fd.project(fd.as_vector([avg+mag*fd.sin(2*fd.pi*(x+shift))]), V)
    np.random.seed(seed)
    if noise is not None:
        for dat in ic.dat:
            dat.data[:] += noise*np.random.random_sample(dat.data.shape)
    return ic


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
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()[0]

if args.show_args:
    Print(args)

# problem parameters
nu = fd.Constant(args.ubar/args.re)
dt = args.cfl/(args.nx*args.ubar)
dt1 = fd.Constant(1/dt)
theta = fd.Constant(args.theta)

mesh = fd.PeriodicUnitIntervalMesh(args.nx)
V = fd.VectorFunctionSpace(mesh, "CG", 2)

un = fd.Function(V, name="Velocity")
un1 = fd.Function(V, name="VelocityNext")

x, = fd.SpatialCoordinate(mesh)
ic = initial(V, avg=args.ubar, mag=0.5, shift=0, noise=None)

un.assign(ic)
un1.assign(ic)


# mass matrix for time derivative
def mass(u, v):
    return fd.inner(u, v)*fd.dx


# spatial gradient terms
def tendency(u, v):
    A = fd.inner(fd.dot(u, fd.nabla_grad(u)), v)
    D = nu*fd.inner(fd.grad(u), fd.grad(v))
    return (A + D)*fd.dx


# implicit theta timestepping forms
v = fd.TestFunction(V)
M = dt1*mass(un1-un, v)
A = theta*tendency(un1, v) + (1-theta)*tendency(un, v)

F = M + A

solver = fd.NonlinearVariationalSolver(
    fd.NonlinearVariationalProblem(F, un1))

# output to non-periodic mesh
V_out = fd.VectorFunctionSpace(fd.UnitIntervalMesh(args.nx), "CG", 1)
write = InterpWriter("output/burgers.pvd", V, V_out, "Velocity").write
write(un, t=0)

t = 0.0
while (t <= args.tend):
    solver.solve()
    un.assign(un1)
    t += dt
    write(un, t=t)
