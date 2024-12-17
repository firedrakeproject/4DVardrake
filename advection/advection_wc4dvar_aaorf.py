import firedrake as fd
from firedrake.__future__ import interpolate
from firedrake.adjoint import (continue_annotation, pause_annotation, stop_annotating,
                               Control, taylor_test, minimize)
from firedrake.adjoint import FourDVarReducedFunctional
from advection_utils import *

control = fd.EnsembleFunction(
    ensemble, [V for _ in range(len(targets))])

for x in control.subfunctions:
    x.assign(background)

ccop = control.copy()
continue_annotation()

# create 4DVar reduced functional and record
# background and initial observation functionals

Jhat = FourDVarReducedFunctional(
    Control(control),
    background_iprod=norm2(B),
    observation_iprod=norm2(R),
    observation_err=observation_error(0),
    weak_constraint=True)

nstep = 0
# record observation stages
with Jhat.recording_stages() as stages:
    # loop over stages
    for stage, ctx in stages:
        # start forward model
        qn.assign(stage.control)

        # propogate
        for _ in range(observation_freq):
            qn1.assign(qn)
            stepper.solve()
            qn.assign(qn1)
            nstep += 1

        obs_index = stage.global_index + 1
        print(f"{obs_index = } | {nstep = } | {fd.norm(qn) = }")

        # secretly save state at obs times
        with stop_annotating():
            ccop.subfunctions[obs_index].assign(qn)

        # take observation
        obs_err = observation_error(obs_index)
        stage.set_observation(qn, obs_err,
                              observation_iprod=norm2(R),
                              forward_model_iprod=norm2(Q))

pause_annotation()

assert len(values) == observation_n+1

vals = control.copy()
for v0, v1 in zip(vals.subfunctions, values):
    v0.assign(v1)

vals1 = control.copy()
for v0, v1 in zip(vals1.subfunctions, values1):
    v0.assign(v1)

control.assign(ccop)
print(f"{[fd.norm(v) for v in values1] = }")
print(f"{[fd.norm(c) for c in control.subfunctions] = }")

print(f"{Jhat(control) = }")
derivatives = Jhat.derivative().subfunctions
print(f"{[fd.norm(d) for d in derivatives] = }")
print()

print(f"{Jhat(vals) = }")
derivatives = Jhat.derivative().subfunctions
print(f"{[fd.norm(d) for d in derivatives] = }")
print()
print(f"{Jhat(vals1) = }")
derivatives = Jhat.derivative().subfunctions
print(f"{[fd.norm(d) for d in derivatives] = }")

print()
print(f"{Jhat(control) = }")
print(f"{taylor_test(Jhat, control, vals) = }")

options = {'disp': True, 'ftol': 1e-2}
derivative_options = {'riesz_representation': None}

opt = minimize(Jhat, options=options, method="L-BFGS-B",
               derivative_options=derivative_options)
