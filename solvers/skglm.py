import numpy as np

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from skglm.datafits import LogisticGroup
    from skglm.penalties import WeightedGroupL2
    from skglm.solvers import GroupBCD, GroupProxNewton


class Solver(BaseSolver):
    name = 'skglm'

    parameters = {
        'solver': [
            'GroupBCD',
            'GroupProxNewton'
        ]
    }

    def __init__(self, solver):
        self.solver = solver
        self.tol = 1e-9

    def set_objective(self, X, y, alpha, grp_ptr, grp_indices):
        self.X, self.y = X, y

        weights = np.ones(X.shape[1])
        self.penalty = WeightedGroupL2(alpha, weights, grp_ptr, grp_indices)
        self.datafit = LogisticGroup(grp_ptr, grp_indices)

        if self.solver == 'GroupBCD':
            self.current_solver = GroupBCD(tol=self.tol, fit_intercept=False)
        elif self.solver == 'GroupProxNewton':
            self.current_solver = GroupProxNewton(
                tol=self.tol, fit_intercept=False)

        # pre compile
        self.run(5)

    def run(self, n_iter):
        self.current_solver.max_iter = n_iter
        self.w = self.current_solver.solve(
            self.X, self.y, self.datafit, self.penalty)[0]

    def get_result(self):
        return self.w.flatten()
