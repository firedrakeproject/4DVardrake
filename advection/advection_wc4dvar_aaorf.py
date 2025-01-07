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

        # take observation
        obs_err = observation_error(stage.observation_index)
        stage.set_observation(qn, obs_err,
                              observation_iprod=norm2(R),
                              forward_model_iprod=norm2(Q))

pause_annotation()

# the perturbation values need to be held in the
# same type as the control i.e. and EnsembleFunction
vals = control.copy()
for v0, v1 in zip(vals.subfunctions, values):
    v0.assign(v1)

print(f"{Jhat(control) = }")
print(f"{taylor_test(Jhat, control, vals) = }")

options = {'disp': True, 'ftol': 1e-2}
derivative_options = {'riesz_representation': 'l2'}

opt = minimize(Jhat, options=options, method="L-BFGS-B",
               derivative_options=derivative_options)
