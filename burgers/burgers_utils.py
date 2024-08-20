import firedrake as fd
from firedrake.__future__ import Interpolator
from firedrake.output import VTKFile
import numpy as np


# interpolate solution to non-periodic mesh to write output
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


def burgers_stepper(nu, dt, theta, mesh, params=None):
    nu = fd.Constant(nu)
    dt1 = fd.Constant(1/dt)
    theta = fd.Constant(theta)

    params = params or {}

    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    un = fd.Function(V, name="Velocity")
    un1 = fd.Function(V, name="VelocityNext")

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

    stepper = fd.NonlinearVariationalSolver(
        fd.NonlinearVariationalProblem(F, un1),
        solver_parameters=params,
        ad_block_tag='Forward model')

    return un, un1, stepper


def noisy_double_sin(V, avg=1, mag=0.5, shift=0, noise=None, seed=123):
    x, = fd.SpatialCoordinate(V.mesh())
    ic = fd.project(fd.as_vector([avg + mag*fd.sin(2*fd.pi*(x+shift))
                                  + 0.2*mag*fd.cos(6*fd.pi*(x-shift))]), V)
    if seed is not None:
        np.random.seed(seed)
    if noise is not None:
        for dat in ic.dat:
            dat.data[:] += noise*np.random.random_sample(dat.data.shape)
    return ic
