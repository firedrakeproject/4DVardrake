import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc

from advection_utils import *

import numpy as np
np.set_printoptions(legacy='1.25')

Print = PETSc.Sys.Print

nx = 50
cfl = 1.6
nt = 40

nprint = 5

u = 1.
dx = 1./nx
dt = cfl*dx/u

mesh = fd.PeriodicUnitIntervalMesh(nx)

V = fd.FunctionSpace(mesh, "DG", 1)

qn, qn1, stepper = timestepper(mesh, V, dt, u)

# initial conditions
x, = fd.SpatialCoordinate(mesh)
ic = fd.Function(V).interpolate(analytic_solution(mesh, u, t=0))
qn.assign(ic)

file = VTKFile('output/advection.pvd', comm=mesh.comm)
file.write(qn, t=0)

t = 0.
for i in range(nt):
    qn1.assign(qn)
    stepper.solve()
    t += dt
    qn.assign(qn1)

    file.write(qn, t=t)

    if (i+1) % nprint == 0:
        Print(f"Timestep {str(i+1).rjust(3)} | t = {str(round(t, 4)).ljust(5)} | norm(qn) = {str(round(fd.norm(qn), 5)).ljust(5)}")
