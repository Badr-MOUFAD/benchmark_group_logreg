from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.datasets import make_correlated_data


class Dataset(BaseDataset):
    name = "simulated"

    parameters = {
        'n_samples, n_features': [
            # (1000, 1000),
            (500, 5000)
        ]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=0):
        self.n_samples, self.n_features = n_samples, n_features
        self.random_state = random_state

    def get_data(self):
        X, y, _ = make_correlated_data(self.n_samples, self.n_features, rho=0.3,
                                       random_state=self.random_state)
        y = np.sign(y)
        return dict(X=X, y=y)
