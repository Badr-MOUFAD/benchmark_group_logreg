from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Objective(BaseObjective):
    name = "Group Logistic regression"

    parameters = {
        'n_groups': []
    }

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.grp_ptr, self.grp_indices = np.array(), np.array()

    def compute(self, beta):
        datafit_val = np.log(1 + np.exp(-self.y * self.X @ beta)).mean()

        penalty_val = 0.
        for g in range(self.n_groups):
            grp_g_indices = self.grp_indices[self.grp_ptr[g]:self.grp_ptr[g+1]]
            penalty_val += norm(beta[grp_g_indices], ord=2)

        return datafit_val + penalty_val

    def to_dict(self):
        return dict(X=self.X, y=self.y, grp_ptr=self.grp_ptr,
                    grp_indices=self.grp_indices)
