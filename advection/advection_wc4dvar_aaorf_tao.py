import firedrake as fd
from firedrake.adjoint import (
    continue_annotation, pause_annotation, minimize,
    stop_annotating, Control, taylor_test)
from firedrake.adjoint import FourDVarReducedFunctional
from advection_utils import *
from fdvar import TAOSolver
from sys import exit

from advection_wc4dvar_aaorf import make_fdvrf
Jhat, control = make_fdvrf()

# the perturbation values need to be held in the
# same type as the control i.e. and EnsembleFunction
vals = control.copy()
for v0, v1 in zip(vals.subfunctions, values):
    v0.assign(v1)

# print(f"{Jhat(control) = }")
# print(f"{taylor_test(Jhat, control, vals) = }")

# options = {'disp': True, 'ftol': 1e-2}
# derivative_options = {'riesz_representation': None}
# opt = minimize(Jhat, options=options, method="L-BFGS-B",
#                derivative_options=derivative_options)
# exit()

ksp_params = {
    'monitor': None,
    'converged_rate': None,
    'rtol': 1e-1,
}

tao_params = {
    'tao_view': ':tao_view.log',
    'tao': {
        'monitor': None,
        'converged_reason': None,
        'gatol': 1e-1,
        'grtol': 1e-1,
        'gttol': 1e-1,
    },
    'tao_type': 'nls',
    'tao_nls': {
        'ksp': ksp_params,
        'ksp_type': 'cg',
        'pc_type': 'lmvm',
    },
    'tao_cg': {
        'ksp': ksp_params,
        'type': 'pr',  # fr-pr-prp-hs-dy
    },
}
tao = TAOSolver(Jhat, options_prefix="",
                solver_parameters=tao_params)
tao.solve()
