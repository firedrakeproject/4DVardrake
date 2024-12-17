import firedrake as fd
from firedrake.__future__ import interpolate
from firedrake.adjoint import (continue_annotation, pause_annotation, stop_annotating,
                               Control, taylor_test, ReducedFunctional, minimize)
from advection_utils import *

# record observation stages

control = background.copy(deepcopy=True)

continue_annotation()

# background functional
J = norm2(B)(control - background)

# initial observation functional
J += norm2(R)(observation_error(0)(control))

nstep = 0
qn.assign(control)

for i in range(1, len(targets)):

    for _ in range(observation_freq):
        qn1.assign(qn)
        stepper.solve()
        qn.assign(qn1)
        nstep += 1
    print(f"{i = } | {nstep = } | {fd.norm(qn) = }")

    # observation functional
    J += norm2(R)(observation_error(i)(qn))

pause_annotation()

Jhat = ReducedFunctional(J, Control(control))

print(f"{Jhat(control) = }")
print(f"{fd.norm(Jhat.derivative()) = }")
print()

print(f"{Jhat(values[0]) = }")
print(f"{fd.norm(Jhat.derivative()) = }")
print()

print(f"{Jhat(values1[0]) = }")
print(f"{fd.norm(Jhat.derivative()) = }")
print()

print(f"{Jhat(control) = }")
print(f"{fd.norm(Jhat.derivative()) = }")
print(f"{taylor_test(Jhat, control, values[0]) = }")

options = {'disp': True, 'ftol': 1e-4}
derivative_options = {'riesz_representation': None}

opt = minimize(Jhat, options=options, method="L-BFGS-B",
               derivative_options=derivative_options)

print(f"{Jhat(opt) = }")
