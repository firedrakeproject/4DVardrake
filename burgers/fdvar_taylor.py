from numpy import mean
from firedrake import assemble, inner, dx, Function, PETSc
from firedrake.adjoint import (
    taylor_to_dict, set_working_tape, ReducedFunctional, Control,
    annotate_tape, continue_annotation, pause_annotation)
from firedrake.adjoint.composite_reduced_functional import CompositeReducedFunctional

Print = PETSc.Sys.Print

def norm4(x):
    """Operation of higher-than-quadratic nonlinearity so the Hessian isn't constant."""
    return assemble(inner(x, x)*dx)**2

def norm4rf(V):
    x = Function(V)
    annotating = annotate_tape()
    if not annotating:
        continue_annotation()
    with set_working_tape() as tape:
        rf = ReducedFunctional(norm4(x), Control(x), tape=tape)
    if not annotating:
        pause_annotation()
    return rf
        

def taylor2(J, m, h, msg, comm, residuals=False):
    """Convenience for running and printing the second order Taylor test."""
    Print(f">>> {msg} >>>", comm=comm)
    taylor = taylor_to_dict(J, m, h)
    if residuals:
        Print(f"{taylor['R0']['Residual'] = }", comm=comm)
        Print(f"{taylor['R1']['Residual'] = }", comm=comm)
        Print(f"{taylor['R2']['Residual'] = }", comm=comm)
    Print(f"{mean(taylor['R0']['Rate']) = }", comm=comm)
    Print(f"{mean(taylor['R1']['Rate']) = }", comm=comm)
    Print(f"{mean(taylor['R2']['Rate']) = }", comm=comm)
    Print('', comm=comm)

def observation_taylor(observation_error,
                       observation_norm,
                       observation_rf,
                       m, h, name, residuals=False):
        rf4 = norm4rf(observation_error.functional.function_space())

        comm = m.function_space().mesh().comm

        Jerror = CompositeReducedFunctional(observation_error, rf4)
        taylor2(Jerror, m, h,
                f'{name} observation error',
                comm, residuals=residuals)

        taylor2(observation_norm,
                observation_error(m), observation_error(h),
                f'{name} observation norm',
                comm, residuals=residuals)

        taylor2(observation_rf, m, h,
                f'{name} observation',
                comm, residuals=residuals)

def model_taylor(forward_model, model_error, model_norm, model_error_rf,
                 ms, hs, name, residuals=False):
    rf4 = norm4rf(ms[0].function_space())

    comm = ms[0].function_space().mesh().comm

    Jforward = CompositeReducedFunctional(forward_model, rf4)
    taylor2(Jforward, ms[0], hs[0],
            f'{name} forward model',
            comm, residuals=residuals)

    Jerror = CompositeReducedFunctional(model_error, rf4)
    taylor2(Jerror, ms, hs,
            f'{name} model error',
            comm, residuals=residuals)

    taylor2(model_norm, ms[1], hs[1],
            f'{name} model norm',
            comm, residuals=residuals)

    taylor2(model_error_rf, ms, hs,
            f'{name} model error-norm rf',
            comm, residuals=residuals)

def fdvar_taylor(Jfdv, m, h, residuals=False,
                 test_fdv=True,
                 test_background=True,
                 test_observations=True,
                 test_model=True,
                 test_stage=True):
    """Run second order Taylor tests on a 4DVar ReducedFunctional and subcomponents."""

    ms = m.subfunctions
    hs = h.subfunctions

    ensemble = Jfdv.ensemble

    if test_fdv:
        taylor2(Jfdv, m, h, 'FourDVarReducedFunctional',
                ensemble.global_comm, residuals=residuals)

    ensemble.ensemble_comm.Barrier()
    if ensemble.ensemble_comm.rank == 0:
        if test_background:
            rf4 = norm4rf(m.subfunctions[0].function_space())
            Jbkg_error = CompositeReducedFunctional(Jfdv.background_error, rf4)
            taylor2(Jbkg_error, ms[0], hs[0],
                    'Background error',
                    ensemble.comm, residuals=residuals)

            taylor2(Jfdv.background_norm, ms[0], hs[0],
                    'Background norm',
                    ensemble.comm, residuals=residuals)

            taylor2(Jfdv.background_rf, ms[0], hs[0],
                    'Background',
                    ensemble.comm, residuals=residuals)
    ensemble.ensemble_comm.Barrier()

    if test_observations:
        with ensemble.sequential(synchronise=True):
            if ensemble.ensemble_comm.rank == 0:
                observation_taylor(
                    Jfdv.initial_observation_error,
                    Jfdv.initial_observation_norm,
                    Jfdv.initial_observation_rf,
                    ms[0], hs[0],
                    'Initial', residuals)

            for i, stage in enumerate(Jfdv.stages):
                observation_taylor(
                    stage.observation_error,
                    stage.observation_norm,
                    stage.observation_rf,
                    ms[i], hs[i],
                    f'Stage {stage.global_index}', residuals)

    if test_model:
        with ensemble.sequential(synchronise=True):
            for i, stage in enumerate(Jfdv.stages):
                mstage = [ms[-1], ms[0]] if i == 0 else ms[i-1:i+1]
                hstage = [hs[-1], hs[0]] if i == 0 else hs[i-1:i+1]
                model_taylor(
                    stage.forward_model,
                    stage.model_error,
                    stage.model_norm,
                    stage.model_error_rf,
                    mstage, hstage,
                    f'Stage {stage.global_index}', residuals)

    if test_stage:
        with ensemble.sequential(synchronise=True):
            for i, stage in enumerate(Jfdv.stages):
                mstage = [ms[-1], ms[0]] if i == 0 else ms[i-1:i+1]
                hstage = [hs[-1], hs[0]] if i == 0 else hs[i-1:i+1]
                taylor2(stage, mstage, hstage,
                        f'Stage {stage.global_index}',
                        ensemble.comm, residuals)
