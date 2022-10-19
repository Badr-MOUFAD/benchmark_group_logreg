from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Objective(BaseObjective):
    name = "Group Logistic regression"

    parameters = {
        'rho': [1e-0, 1e-1, 1e-2, 5*1e-3],
    }

    def __init__(self, rho):
        self.rho = rho

    def set_data(self, X, y, grp_ptr, grp_indices):
        self.X, self.y = X, y
        self.grp_ptr, self.grp_indices = grp_ptr, grp_indices

        self.alpha = self.rho * Objective._compute_alpha_max(
            self.X, self.y, self.grp_ptr, self.grp_indices)

    def compute(self, beta):
        n_groups = len(self.grp_ptr) - 1

        penalty_val = 0.
        for g in range(n_groups):
            grp_g_indices = self.grp_indices[self.grp_ptr[g]:self.grp_ptr[g+1]]
            penalty_val += self.alpha * norm(beta[grp_g_indices], ord=2)

        datafit_val = np.mean(np.log(1 + np.exp(-self.y * (self.X @ beta))))

        return datafit_val + penalty_val

    def to_dict(self):
        return dict(X=self.X, y=self.y, alpha=self.alpha, grp_ptr=self.grp_ptr,
                    grp_indices=self.grp_indices)

    @staticmethod
    def _compute_alpha_max(X, y, grp_ptr, grp_indices):
        alpha_max = 0.
        n_groups = len(grp_ptr) - 1
        n_samples = len(y)

        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            alpha_max = max(
                alpha_max,
                norm(X[:, grp_g_indices].T @ y, ord=2) / (2 * n_samples)
            )

        return alpha_max
