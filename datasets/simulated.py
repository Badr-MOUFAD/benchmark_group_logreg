from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.datasets import make_correlated_data


class Dataset(BaseDataset):
    name = "simulated"

    parameters = {
        'n_samples, n_features, n_groups': [
            (500, 5000, 100),
            (500, 5000, 500),
            (500, 5000, 1000),
        ]
    }

    def __init__(self, n_samples=10, n_features=50, n_groups=10, random_state=0):
        self.n_samples, self.n_features = n_samples, n_features
        self.n_groups = n_groups
        self.random_state = random_state

    def get_data(self):
        X, y, _ = make_correlated_data(self.n_samples, self.n_features, rho=0.3,
                                       random_state=self.random_state)

        y = np.sign(y)
        grp_indices, grp_ptr, _ = Dataset._generate_random_grp(
            self.n_groups, self.n_features, shuffle=False)

        return dict(X=X, y=y, grp_ptr=grp_ptr, grp_indices=grp_indices)

    @staticmethod
    def _generate_random_grp(n_groups, n_features, shuffle=False):
        grp_indices = np.arange(n_features, dtype=np.int32)
        np.random.seed(0)
        if shuffle:
            np.random.shuffle(grp_indices)
        np.random.seed(0)
        splits = np.random.choice(
            n_features, size=n_groups+1, replace=False).astype(np.int32)
        splits.sort()
        splits[0], splits[-1] = 0, n_features

        groups = [list(grp_indices[splits[i]: splits[i+1]])
                  for i in range(n_groups)]

        return grp_indices, splits, groups
