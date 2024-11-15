import firedrake as fd
from firedrake.petsc import PETSc
import argparse
from burgers_utils import InterpWriter, burgers_stepper, noisy_double_sin
from math import ceil
import numpy as np

Print = PETSc.Sys.Print

verbose = False

def cprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)


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
parser.add_argument('--nchunks', type=int, default=1, help='Number of chunks in time.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()[0]

if args.show_args:
    Print(args)

# problem parameters
nu = args.ubar/args.re
dt = args.cfl/(args.nx*args.ubar)
theta = args.theta

global_comm = fd.COMM_WORLD
if global_comm.size % args.nchunks != 0:
    raise ValueError("Number of time-chunks must exactly divide size of COMM_WORLD")
nranks_space = global_comm.size // args.nchunks
ensemble = fd.Ensemble(global_comm, nranks_space)
last_rank = ensemble.ensemble_comm.size - 1
trank = ensemble.ensemble_comm.rank

# time-partition
# total number of timesteps
Nt = ceil(args.tend/dt)
# chunk-local number of timesteps
nt = ceil(Nt/args.nchunks)
# last chunk cleans up remainder
if trank == last_rank:
    nt = Nt - nt*(args.nchunks - 1)
ntloc = nt*np.ones(1, dtype=int)
ntglb = np.zeros(1, dtype=int)
ensemble.ensemble_comm.Allreduce(ntloc, ntglb)
assert ntglb[0] == Nt

mesh = fd.PeriodicUnitIntervalMesh(args.nx, comm=ensemble.comm)

un, un1, stepper = burgers_stepper(nu, dt, theta, mesh)
V = un.function_space()

# if ranks == 0 initialise
if trank == 0:
    ic = noisy_double_sin(V, avg=args.ubar, mag=0.5,
                          shift=args.phase_shift,
                          noise=args.noise)
    un.assign(ic)
    un1.assign(ic)

# all ranks create output with id

# output to non-periodic mesh
V_out = fd.VectorFunctionSpace(fd.UnitIntervalMesh(args.nx, comm=ensemble.comm), "CG", 1)
writer = InterpWriter(f"output/burgers_chunk{str(trank).rjust(3, '0')}.pvd", V, V_out, "Velocity")

# if rank != 0, wait for recv of (solution, time)
cprint(f"@ 0: {trank = }")
if trank == 0:
    t = 0
    writer.write(un, t=t)
else:
    ensemble.recv(un, source=trank-1, tag=trank+000)
    ttmp = np.zeros(1)
    ensemble.ensemble_comm.Recv(ttmp, source=trank-1, tag=trank+100)
    t = ttmp[0]
cprint(f"@ 1: {trank = }")

# ranks loop for nt and pass on solution and time
cprint(f"{trank=} | {nt = }")
for i in range(nt):
    cprint(f"{trank=} | {i = }")
    un1.assign(un)
    cprint(f"{trank=} | a")
    stepper.solve()
    cprint(f"{trank=} | b")
    un.assign(un1)
    t += dt
    cprint(f"{trank=} | c")
    writer.write(un, t=t)
    cprint(f"{trank=} | d")
    Print(f"Time = {str(round(t, 4)).ljust(6)} | Norm = {fd.norm(un)}", comm=ensemble.comm)
cprint(f"@ 2: {trank = }")

if trank != last_rank:
    ensemble.send(un, dest=trank+1, tag=trank+1+000)
    ensemble.ensemble_comm.Send(t*np.ones(1), dest=trank+1, tag=trank+1+100)
cprint(f"@ 3: {trank = }")
