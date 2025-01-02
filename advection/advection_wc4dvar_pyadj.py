import firedrake as fd
from firedrake.__future__ import interpolate
from firedrake.adjoint import (continue_annotation, pause_annotation, stop_annotating,
                               Control, taylor_test, ReducedFunctional, minimize)
from advection_utils import *

# record observation stages

control = [background.copy(deepcopy=True)
            for _ in range(len(targets))]

continue_annotation()

# background functional
J = norm2(B)(control[0] - background)

# initial observation functional
J += norm2(R)(observation_error(0)(control[0]))

nstep = 0
for i in range(1, len(control)):
    # start from previous control
    qn.assign(control[i-1])

    for _ in range(observation_freq):
        qn1.assign(qn)
        stepper.solve()
        qn.assign(qn1)
        nstep += 1

    # smuggle previous state over the observation time
    with stop_annotating():
        control[i].assign(qn)

    # model error functional
    J += norm2(Q)(qn - control[i])

    # observation functional
    J += norm2(R)(observation_error(i)(control[i]))

pause_annotation()

Jhat = ReducedFunctional(J, [Control(c) for c in control])

print(f"{Jhat(control) = }")
print(f"{taylor_test(Jhat, control, values) = }")

options = {'disp': True, 'ftol': 1e-2}
derivative_options = {'riesz_representation': 'l2'}

opt = minimize(Jhat, options=options, method="L-BFGS-B",
               derivative_options=derivative_options)
