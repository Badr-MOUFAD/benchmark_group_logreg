# yngvem uses cross entropy as a loss
# cf. https://github.com/yngvem/group-lasso/blob/aa1c77aa3ba8534d543e3eec6b6f522d7996e2bf/src/group_lasso/_group_lasso.py#L859-L860
# hence, it can't be compared to skglm GroupLogisticRegression
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import warnings
    from group_lasso import LogisticGroupLasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'yngvem'

    def __init__(self):
        self.tol = 1e-9

    def set_objective(self, X, y, alpha, grp_ptr, grp_indices):
        self.X, self.y = X, y

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        yn_groups = Solver._convert_to_group_yngvem(grp_ptr, grp_indices)
        self.estimator = LogisticGroupLasso(
            groups=yn_groups,
            group_reg=alpha,  # without weights
            l1_reg=0.,
            subsampling_scheme=None,
            fit_intercept=False,
            scale_reg="none",
            supress_warning=True,
            n_iter=1_000,
            tol=self.tol,
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1])
            return

        self.estimator.n_iter = n_iter
        self.estimator.fit(self.X, self.y)
        yn_coef = self.estimator.coef_

        # weird cf. section Visualise regression coefficients
        # https://group-lasso.readthedocs.io/en/latest/auto_examples/example_logistic_group_lasso.html#visualise-regression-coefficients
        self.coef = yn_coef[:, 1] - yn_coef[:, 0]

    def get_result(self):
        return self.coef.flatten()

    @staticmethod
    def _convert_to_group_yngvem(grp_ptr, grp_indices):
        # each feature (index of arr) is assigned the group to which its belong
        # cf. docs: https://group-lasso.readthedocs.io/en/latest/api_reference.html
        n_groups, n_features = len(grp_ptr) - 1, len(grp_indices)

        groups = np.zeros(n_features)
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
            groups[grp_g_indices] = g + 1
