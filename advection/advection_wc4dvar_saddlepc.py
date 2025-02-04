import firedrake as fd
from firedrake.petsc import PETSc, OptionsManager
from firedrake.adjoint import pyadjoint  # noqa: F401
from pyadjoint.optimization.tao_solver import PETScVecInterface
from advection_wc4dvar_aaorf import make_fdvrf
from mpi4py import MPI
import numpy as np
from typing import Optional


# CovarianceNormRF Mat
def CovarianceMat(covariancerf):
    covariance = covariancerf.covariance
    space = covariancerf.controls[0].control.function_space()
    comm = space.mesh().comm
    sizes = space.dof_dset.layout_vec.sizes
    shape = (sizes, sizes)
    covmat = PETSc.Mat().createConstantDiagonal(
        shape, covariance, comm=comm)
    covmat.setUp()
    covmat.assemble()
    return covmat


# pyadjoint RF Mat
class ReducedFunctionalMatCtx:
    """
    PythonMat context to apply action of a pyadjoint.ReducedFunctional.

    Parameters
    ----------

        action_type
            Union['hessian', 'tlm', 'adjoint']
    """
    def __init__(self, Jhat: pyadjoint.ReducedFunctional,
                 action_type: str = 'hessian',
                 derivative_options: Optional[dict] = None,
                 comm: MPI.Comm = PETSc.COMM_WORLD):
        self.Jhat = Jhat
        self.control_interface = PETScVecInterface(Jhat.controls, comm=comm)
        self.functional_interface = PETScVecInterface(
            Jhat.functional, comm=comm)

        if action_type == 'hessian':
            self.xinterface = self.control_interface
            self.yinterface = self.control_interface
        elif action_type == 'adjoint':
            self.xinterface = self.functional_interface
            self.yinterface = self.control_interface
        elif action_type == 'tlm':
            self.xinterface = self.control_interface
            self.yinterface = self.functional_interface
        else:
            raise ValueError(
                'Unrecognised {action_type = }.')

        self.action_type = action_type
        self._m = Jhat.control.copy()
        self.derivative_options = derivative_options
        self._shift = 0

        self.default_mult = {
            'hessian': self._mult_hessian,
            'tlm': self._mult_tlm,
            'adjoint': self._mult_adjoint
        }[action_type]

        # Storage for result of action.
        # Possibly in the dual space for adjoint actions.
        if action_type == 'adjoint':
            self.Jhat(self._m)
            self._mdot = self.Jhat(
                derivative_options=derivative_options)
        else:
            self._mdot = Jhat.control.copy()

    @classmethod
    def update(cls, obj, x, A, P):
        ctx = A.getPythonContext()
        ctx.control_interface.from_petsc(x, ctx._m)
        ctx._shift = 0

    def update_tape_values(self, update_adjoint=True):
        _ = self.Jhat(self._m)
        if update_adjoint:
            _ = self.Jhat.derivative(options=self.derivative_options)

    def mult(self, A, x, y):
        self.xinterface.from_petsc(x, self._mdot)
        out = self.default_mult(A, self._mdot)
        self.yinterface.to_petsc(out, y)
        if self._shift != 0:
            y.axpy(self._shift, x)

    def _mult_hessian(self, A, x):
        if self.action_type != 'hessian':
            raise NotImplementedError(
                f'Cannot apply hessian action if {self.action_type = }')
        self.update_tape_values()
        return self.Jhat.hessian(x)

    def _mult_tlm(self, A, x):
        if self.action_type != 'tlm':
            raise NotImplementedError(
                f'Cannot apply tlm action if {self.action_type = }')
        self.update_tape_values(update_adjoint=False)
        return self.Jhat.tlm(x)

    def _mult_adjoint(self, A, x):
        if self.action_type != 'adjoint':
            raise NotImplementedError(
                f'Cannot apply adjoint action if {self.action_type = }')
        self.update_tape_values(update_adjoint=False)
        return self.Jhat.derivative(
            adj_value=x, derivative_options=self.derivative_options)


def ReducedFunctionalMat(self, Jhat, action_type='hessian',
                         derivative_options=None,
                         comm=PETSc.COMM_WORLD):
    ctx = ReducedFunctionalMatCtx(
        Jhat, action_type,
        derivative_options,
        comm=comm)
    # TODO: use functional_interface sizes
    #       to allow non-square matrices.
    n = ctx.control_interface.n
    N = ctx.control_interface.N
    mat = PETSc.Mat().createPython(
        ((n, N), (n, N)), ctx, comm=comm)
    mat.setUp()
    mat.assemble()
    return mat


class EnsembleBlockDiagonalMat:
    def __init__(self, ensemble, spaces, blocks):
        if isinstance(spaces, fd.EnsembleFunction):
            if spaces.ensemble is not ensemble:
                raise ValueError(
                    "Ensemble of EnsembleFunction must match ensemble provided")
            spaces = spaces.local_function_spaces
        if len(blocks) != len(spaces):
            raise ValueError(
                f"EnsembleBlockDiagonalMat requires one submatrix for each of the"
                f" {len(spaces)} local subfunctions of theEnsembleFunction, but"
                f" only {len(blocks)} provided.")

        for i, (subspace, block) in enumerate(zip(spaces, blocks)):
            vsizes = subspace.dof_dset.layout_vec.sizes
            msizes = block.sizes
            if msizes[0] != msizes[1]:
                raise ValueError(
                    f"Block {i} of EnsembleBlockDiagonalMat must be square, not {msizes}")
            if msizes[0] != vsizes:
                raise ValueError(
                    f"Block {i} of EnsembleBlockDiagonalMat must have shape {(vsizes, vsizes)}"
                    f" to match the EnsembleFunction, not shape {msizes}")

        self.ensemble = ensemble
        self.blocks = blocks
        self.spaces = spaces

        # EnsembleFunction knows how to split out subvecs for each block
        self.x = fd.EnsembleFunction(self.ensemble, spaces)
        self.y = fd.EnsembleCofunction(self.ensemble, [V.dual() for V in spaces])

    def mult(self, A, x, y):
        with self.x.vec_wo() as xvec:
            x.copy(xvec)

        # compute answer
        subvecs = zip(self.x.subfunctions, self.y.subfunctions)
        for block, (xsub, ysub) in zip(self.blocks, subvecs):
            with xsub.dat.vec_ro as xvec, ysub.dat.vec_wo as yvec:
                block.mult(xvec, yvec)

        with self.y.vec_ro() as yvec:
            yvec.copy(y)


class EnsembleBlockDiagonalPC:
    prefix = "ensemblejacobi_"

    def __init__(self):
        self.initialized = False

    def setUp(self, pc):
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        pcprefix = pc.getOptionsPrefix()
        prefix = pcprefix + self.prefix
        options = PETSc.Options(prefix)

        _, P = pc.getOperators()
        ensemble_mat = P.getPythonContext()
        ensemble = ensemble_mat.ensemble
        spaces = ensemble_mat.spaces
        submats = ensemble_mat.blocks

        self.ensemble = ensemble
        self.spaces = spaces
        self.submats = submats

        self.x = fd.EnsembleFunction(ensemble, spaces)
        self.y = fd.EnsembleCofunction(ensemble,
                                       [V.dual() for V in spaces])

        subksps = []
        for i, mat in enumerate(submats):
            ksp = PETSc.KSP().create(comm=ensemble.comm)
            ksp.setOperators(mat)

            sub_prefix = pcprefix + f"sub_{i}_"
            # TODO: default options
            options = OptionsManager({}, sub_prefix)
            options.set_from_options(ksp)
            self.subksps.append((ksp, options))

        self.subksps = tuple(subksps)

    def apply(self, pc, x, y):
        with self.x.vec_wo() as xvec:
            x.copy(xvec)

        subfuncs = zip(self.x.subfunctions, self.y.subfunctions)
        for (subksp, suboptions), (subx, suby) in zip(self.subksps, subfuncs):
            with subx.dat.vec_ro as rhs, suby.dat.vec_wo as sol:
                with suboptions.inserted_options():
                    subksp.solve(rhs, sol)

        with self.y.vec_ro() as yvec:
            yvec.copy(y)


class EnsembleMat:
    def __init__(self, ensemblefunction, ctx):
        self.ensemble = ensemblefunction.ensemble
        sizes = ensemblefunction._vec.sizes
        self.petsc_mat = PETSc.Mat().createPython(
            (sizes, sizes), ctx,
            comm=self.ensemble.comm)
        self.petsc_mat.setUp()
        self.petsc_mat.assemble()


Jhat, control = make_fdvrf()
ensemble = Jhat.ensemble

# >>>>> Covariance

# Covariance Mat
# covrf = Jhat.background_norm
covrf = Jhat.stages[0].model_norm
covmat = CovarianceMat(covrf)

# Covariance KSP
covksp = PETSc.KSP().create(comm=ensemble.comm)
covksp.setOptionsPrefix('cov_')
covksp.setOperators(covmat)

covksp.pc.setType(PETSc.PC.Type.JACOBI)
covksp.setType(PETSc.KSP.Type.PREONLY)
covksp.setFromOptions()
covksp.setUp()
print(PETSc.Options().getAll())

x = covmat.createVecRight()
b = covmat.createVecLeft()

b.array_w[:] = np.random.random_sample(b.array_w.shape)
print(f'{b.norm() = }')
covksp.solve(b, x)
print(f'{x.norm() = }')
