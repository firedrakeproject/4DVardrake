import firedrake as fd
from firedrake.__future__ import interpolate
from firedrake.adjoint import (continue_annotation, pause_annotation,
                               stop_annotating, Control, taylor_test,
                               ReducedFunctional, minimize)
from firedrake.adjoint import FourDVarReducedFunctional
from advection_utils import *

control = background.copy(deepcopy=True)

continue_annotation()

# create 4DVar reduced functional and record
# background and initial observation functionals

Jhat = FourDVarReducedFunctional(
    Control(control),
    background_iprod=norm2(B),
    observation_iprod=norm2(R),
    observation_err=observation_error(0),
    weak_constraint=False)

nstep = 0
# record observation stages
with Jhat.recording_stages(nstages=len(targets)-1) as stages:
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
                              observation_iprod=norm2(R))

pause_annotation()

print(f"{taylor_test(Jhat, control, values[0]) = }")

options = {'disp': True, 'ftol': 1e-2}
derivative_options = {'riesz_representation': None}

opt = minimize(Jhat, options=options, method="L-BFGS-B",
               derivative_options=derivative_options)

print(f"{fd.errornorm(targets[0], control) = }")
print(f"{fd.errornorm(targets[0], opt) = }")
