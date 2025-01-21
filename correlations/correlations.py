from firedrake.petsc import PETSc
from scipy import sparse
import numpy as np


def minmag(a, b):
    return np.where(np.abs(a) < np.abs(b), a, b)


def periodic_separation(y, x, start=None, end=None):
    start = x[0] if start is None else start
    end = x[-1] if end is None else end
    internal = x - y
    left = (y - start) + (end - x)
    right = (end - y) + x
    return minmag(minmag(left, internal), right)


def chordal_distance(separation, circumference):
    diameter = circumference/np.pi
    angle = 2*np.pi*separation/circumference
    return diameter*np.sin(angle/2)


def chordal_separation(y, x, start=None, end=None):
    start = x[0] if start is None else start
    end = x[-1] if end is None else end
    return chordal_distance(
        periodic_separation(y, x, start, end),
        end - start)


def separation_csr(x, maxsep=None, separation=None,
                   triangular=True):
    n = len(x)
    upper = sparse.csr_array((n, n))
    for i, y in enumerate(x):
        if separation:
            d = separation(y, x[i:])
        else:
            d = x[i:] - y
        if maxsep:
            d[d > maxsep] = 0
        upper[i, i:] = d
    upper.eliminate_zeros()
    upper.setdiag(0)
    if triangular:
        return upper
    else:
        full = symmetrise_csr(upper)
        full.setdiag(0)
        return full


def symmetrise_csr(csr):
    transpose = csr.T.tocsr()
    transpose.setdiag(0)
    return csr + transpose


def soar(r, L, sigma=1.):
    rL = np.abs(r)/L
    return sigma*(1 + rL)*np.exp(-rL)


def soar_csr(x, L, sigma=1., tol=None,
             maxsep=None, separation=None,
             triangular=True):
    upper = separation_csr(x, maxsep=maxsep,
                           separation=separation,
                           triangular=True)
    upper.data[:] = soar(upper.data, L, sigma=sigma)
    if tol:
        upper.data[upper.data < tol] = 0
    upper.eliminate_zeros()
    return upper if triangular else symmetrise_csr(upper)


def csr_to_petsc(scipy_mat):
    mat = PETSc.Mat().create()
    mat.setType('aij')
    mat.setSizes(scipy_mat.shape)
    mat.setValuesCSR(scipy_mat.indptr,
                     scipy_mat.indices,
                     scipy_mat.data)
    mat.assemble()
    return mat


def petsc_to_csr(petsc_mat):
    return sparse.csr_matrix(petsc_mat.getValuesCSR()[::-1],
                             shape=petsc_mat.getSize())
