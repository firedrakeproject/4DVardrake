import firedrake as fd
from firedrake.external_operators import AbstractExternalOperator, assemble_method


class LinearSolverOperator(AbstractExternalOperator):
    def __init__(self, *operands, function_space,
                 derivatives=None, argument_slots=(),
                 operator_data=None):

        AbstractExternalOperator.__init__(
            self, *operands,
            function_space=function_space,
            derivatives=derivatives,
            argument_slots=argument_slots,
            operator_data=operator_data)

        self.b, = operands
        self._b = self.b.copy()
        self.bexpr = fd.inner(self.b, fd.TestFunction(self._b.function_space()))*fd.dx

        self.A = operator_data["A"]
        self.solver = fd.LinearSolver(
            self.A, **operator_data.get("solver_kwargs", {}))

    def _solve(self, b):
        x = fd.Function(self.function_space())
        self.solver.solve(x, b)
        return x

    def _solve_transpose(self, b):
        x = fd.Cofunction(self.function_space.dual())
        self.solver.solve_transpose(x, b)
        return x

    def _assemble_b(self, b):
        self._b.assign(b)
        return fd.assemble(self.bexpr)

    @assemble_method(0, (0,))
    def assemble_operator(self, *args, **kwargs):
        return self._solve(self._assemble_b(self.b))

    @assemble_method(1, (0, None))
    def assemble_jacobian_action(self, *args, **kwargs):
        return self._solve(self._assemble_b(self.argument_slots()[-1]))

    @assemble_method(1, (1, None))
    def assemble_jacobian_adjoint_action(self, *args, **kwargs):
        return self._assemble_b(self._solve_transpose(self.argument_slots()[0]))
