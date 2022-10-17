from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Objective(BaseObjective):
    name = "Group Logistic regression"

    parameters = {
        'rho': [1e-0, 1e-1, 1e-2, 1e-3],
        'n_groups': [50, 100, 500]
    }

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])

    def set_data(self, X, y):
        self.X, self.y = X, y

        self.grp_ptr, self.grp_indices = Objective._generate_random_grp(
            self.n_groups, X.shape[1], shuffle=False)

        self.alpha = Objective._compute_alpha_max(
            X, y, self.grp_ptr, self.grp_indices)

    def compute(self, beta):
        datafit_val = np.log(1 + np.exp(-self.y * self.X @ beta)).mean()

        penalty_val = 0.
        for g in range(self.n_groups):
            grp_g_indices = self.grp_indices[self.grp_ptr[g]:self.grp_ptr[g+1]]
            penalty_val += norm(beta[grp_g_indices], ord=2)

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
            grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
            alpha_max = max(
                alpha_max,
                norm(X[:, grp_g_indices].T @ y, ord=np.inf) / (2 * n_samples)
            )

        return alpha_max

    @staticmethod
    def _generate_random_grp(n_groups, n_features, shuffle=True):
        grp_indices = np.arange(n_features, dtype=np.int32)
        np.random.seed(0)
        if shuffle:
            np.random.shuffle(grp_indices)
        splits = np.random.choice(
            n_features, size=n_groups+1, replace=False).astype(np.int32)
        splits.sort()
        splits[0], splits[-1] = 0, n_features

        groups = [list(grp_indices[splits[i]: splits[i+1]])
                  for i in range(n_groups)]

        return grp_indices, splits, groups
