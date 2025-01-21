import firedrake as fd
from firedrake.adjoint import (
    continue_annotation, pause_annotation, Control, ReducedFunctional)
import numpy as np
import scipy.sparse.linalg as spla
from functools import partial
from correlations import (
    chordal_separation, soar_csr, csr_to_petsc, petsc_to_csr)

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def weighted_norm(M, x, **kwargs):
    y = fd.Function(x.function_space().dual())
    fd.solve(M, y, x, **kwargs)
    return fd.assemble(fd.inner(y, y)*fd.dx)


parser = ArgumentParser(
    description='Construct and solve an SOAR correlation matrix on a periodic interval.',  # noqa: E501
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument('--n', type=int, default=128, help='Number of mesh nodes.')  # noqa: E501
parser.add_argument('--L', type=float, default=0.6, help='Correlation length scale.')  # noqa: E501
parser.add_argument('--sigma', type=float, default=0.4, help='SOAR scaling.')  # noqa: E501
parser.add_argument('--minval', type=float, default=0.1, help='Minimum correlation value. Any below value this will be clipped to zero.')  # noqa: E501
parser.add_argument('--maxnz', type=int, default=16, help='Maximum number of nonzeros per row. Any values beyond this bandwidth will be clipped to zero.')  # noqa: E501
parser.add_argument('--psi', type=float, default=0.1, help='Shift to add to the major diagonal if the matrix is not positive-definite.')  # noqa: E501
parser.add_argument('--neigs', type=int, default=8, help='Number of eigenvalues to calculate to estimate positive-definiteness.')  # noqa: E501
parser.add_argument('--show_args', action='store_true', help='Print all the arguments when the script starts.')  # noqa: E501

args, _ = parser.parse_known_args()

if args.show_args:
    print(args)

np.random.seed(42)

np.set_printoptions(legacy='1.25', precision=3,
                    threshold=2000, linewidth=200)

# number of nodes
n = args.n
maxsep = 0.5*args.maxnz/n

# eigenvalue estimates

mesh = fd.PeriodicUnitIntervalMesh(n)
V = fd.FunctionSpace(mesh, 'DG', 0)
coords = fd.Function(V).interpolate(fd.SpatialCoordinate(mesh)[0])
x = coords.dat.data

chord_sep = partial(chordal_separation,
                    start=0, end=1)

print('>>> Building SOAR CSR matrix')
Bcsr = soar_csr(x, args.L, sigma=args.sigma,
                tol=args.minval, maxsep=maxsep,
                separation=chord_sep,
                triangular=False)
size = n*n
nnz = Bcsr.nnz
nnzrow = nnz/n
fill = nnz/size
print(f'{size = } | {nnz = } | {nnzrow = } | {fill = }')

print('>>> Checking positive-definiteness')
neigs = min(args.neigs, n-2)
emax = np.max(spla.eigsh(Bcsr, k=neigs, which='LM',
                         return_eigenvectors=False))
emin = np.min(spla.eigsh(Bcsr, k=neigs, which='SA',
                         return_eigenvectors=False))
cond = emax/emin
print('Eigenvalues:')
print(f'{emax = } | {emin = } | {cond = }')

Bmat = csr_to_petsc(Bcsr)

if emin < 0:
    print('>>> Matrix is not SPD: shifting...')
    Bmat.shift(abs(emin) + args.psi)
    Bcsr = petsc_to_csr(Bmat)
    emax = np.max(spla.eigsh(Bcsr, k=neigs, which='LM',
                             return_eigenvectors=False))
    emin = np.min(spla.eigsh(Bcsr, k=neigs, which='SA',
                             return_eigenvectors=False))
    cond = emax/emin
    print('Eigenvalues after shift:')
    print(f'{emax = } | {emin = } | {cond = }')

params = {
    'ksp_converged_rate': None,
    'ksp_monitor': None,

    'ksp_rtol': 1e-5,
    'ksp_type': 'cg',
    'pc_type': 'icc',

    # 'ksp_type': 'preonly',
    # 'pc_type': 'lu',
    # 'pc_factor_mat_solver_type': 'mumps',
}

print('>>> Setting up solver')
arguments = (fd.TestFunction(V), fd.TrialFunction(V))

B = fd.AssembledMatrix(arguments, bcs=[], petscmat=Bmat)

x0 = fd.Function(V)
x1 = fd.Function(V)

b0 = fd.Cofunction(V.dual())
b0.dat.data[:] = np.random.random_sample(b0.dat.data.shape)

b1 = fd.Cofunction(V.dual())
b1.dat.data[:] = np.random.random_sample(b1.dat.data.shape)

print('>>> Solving LinearSolver...')
solver = fd.LinearSolver(B, options_prefix='',
                         solver_parameters=params)
solver.solve(x0, b0)
xBx = fd.assemble(fd.inner(b0.riesz_representation(), x0)*fd.dx)
print(f'{xBx = }')

print('>>> Solving LinearSolverOperator...')

from solver_external_operator import LinearSolverOperator
bfunc = fd.Function(V)
solve_operator = LinearSolverOperator(
    bfunc, function_space=V,
    operator_data={
        "A": B,
        "solver_kwargs": {
            "options_prefix": '',
            "solver_parameters": params
        }
    })

bfunc.assign(b0.riesz_representation())
xBx = fd.assemble(fd.inner(b0.riesz_representation(), solve_operator)*fd.dx)
print(f'{xBx = }')

bfunc.assign(b0.riesz_representation())
continue_annotation()
J = fd.assemble(fd.inner(bfunc, solve_operator)*fd.dx)
Jhat = ReducedFunctional(J, Control(bfunc))
pause_annotation()

print(f'{Jhat(b1.riesz_representation()) = }')
print(f'{fd.norm(Jhat.derivative()) = }')

solver.solve(x1, b1)
xBx = fd.assemble(fd.inner(b1.riesz_representation(), x1)*fd.dx)
print(f'{xBx = }')
