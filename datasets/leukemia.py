from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelBinarizer


class Dataset(BaseDataset):
    name = "leukemia"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    parameters = {
        'n_groups': [500, 2000]
    }

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def get_data(self):
        # Unlike libsvm[leukemia], this dataset corresponds to the whole
        # leukemia  data with train + test data (72 samples) and not just
        # the training set.
        X, y = fetch_openml("leukemia", return_X_y=True)
        X = X.to_numpy()
        binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        y = binarizer.fit_transform(y)[:, 0].astype(X.dtype)

        grp_indices, grp_ptr, _ = Dataset._generate_random_grp(
            self.n_groups, X.shape[1], shuffle=False)

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
