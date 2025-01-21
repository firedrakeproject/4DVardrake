import numpy as np
from functools import partial
from correlations import (
    chordal_separation, soar_csr, csr_to_petsc, petsc_to_csr)

np.set_printoptions(legacy='1.25', precision=2,
                    linewidth=200, threshold=10000)

n = 16

L = 0.1
tol = 0.4

maxsep = 0.3

width = 1
start = 0
end = start + width

x = np.linspace(start, end, n, endpoint=False)

chordsep = partial(chordal_separation,
                   start=start, end=end)

petsc_mat = csr_to_petsc(
    soar_csr(x, L, tol=tol, maxsep=maxsep,
             separation=chordsep,
             triangular=False))

print(f'mat =\n{petsc_to_csr(petsc_mat).todense()}')
