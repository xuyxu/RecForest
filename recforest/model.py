"""Implementation of RecForest for Anomaly Detection."""

# Author: Yi-Xuan Xu


import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomTreesEmbedding

from . import _transform as _CLIB


def _parallel_transform(X, tree, tree_idx, amin, amax):
    """
    Private function used to find bounding boxes for each tree in parallel."""

    # Prepare data passed to the C side
    n_samples, _ = X.shape
    X_leaves = tree.apply(X)
    decision_path = tree.decision_path(X)
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    # Generate initial bounding boxes
    lower_bound = np.repeat(amin, n_samples, axis=0)
    upper_bound = np.repeat(amax, n_samples, axis=0)

    # Transform
    _CLIB._transform(X,
                     X_leaves,
                     decision_path,
                     feature,
                     threshold,
                     lower_bound,
                     upper_bound)

    return lower_bound, upper_bound


class RecForest(object):
    """
    Implementation of RecForest for Anomaly Detection.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of decision trees in the forest.
    max_depth : int, default=None
        The maximum depth of decision trees in the forest. ``None`` means no
        limitation on the maximum tree depth.
    n_jobs : int, default=None
        The number of jobs to run in parallel for both `fit` and `transform`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
    random_state : int or None, default=None
        - If ``int``, ``random_state`` is the seed used by the internal random
          number generator;
        - If ``None``, the random number generator is the RandomState
          instance used by `np.random`.

    Attributes
    ----------
    estimator_ : RandomTreesEmbedding
        The backbone model of RecForest.
    """

    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 n_jobs=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.estimator_ = RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=random_state)

    def _rec_error(self, X, X_rec):
        """
        Compute the reconstruction error given the original sample and the
        reconstructed sample.
        """
        assert X.shape == X_rec.shape
        rec_error = np.sum(np.square(X - X_rec), axis=1)

        return rec_error

    def _init_bound(self, X):
        """Initialize the bounding boxes."""
        n_samples, _ = X.shape
        lower_bound = np.repeat(self.amin, n_samples, axis=0)
        upper_bound = np.repeat(self.amax, n_samples, axis=0)

        return lower_bound, upper_bound

    def _transform(self, X):
        """Generate reconstructed samples from the bounding boxes."""
        rets = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_transform)(X, tree, idx, self.amin, self.amax)
            for idx, tree in enumerate(self.estimator_.estimators_))

        # Merge results from workers
        lower_bound, upper_bound = self._init_bound(X)
        for tree_lower, tree_upper in rets:
            lower_bound = np.maximum(lower_bound, tree_lower)
            upper_bound = np.minimum(upper_bound, tree_upper)

        X_rec = .5 * lower_bound + .5 * upper_bound

        return X_rec

    def fit(self, X):
        """
        Build the RecForest from the training set X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input training data.
        """

        # C-aligned
        if not X.flags["C_CONTIGUOUS"]:
            X = np.ascontiguousarray(X)

        self.amax = np.amax(X, axis=0).reshape(1, -1)
        self.amin = np.amin(X, axis=0).reshape(1, -1)
        self.estimator_.fit(X)

        return self

    def apply(self, X):
        """
        Return the leaf node ID for each sample in each decision tree of the
        RecForest.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            The leaf node ID mat, with the i-th row corresponding to the
            leaf node of i-th sample across all decision trees.
        """
        X_leaves = self.estimator_.apply(X)
        return X_leaves

    def predict(self, X):
        """
        Predict raw anomaly scores of X using the fitted RecForest.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly scores of each sample in X.
        """

        # C-aligned
        if not X.flags["C_CONTIGUOUS"]:
            X = np.ascontiguousarray(X)

        n_samples, _ = X.shape
        scores = np.zeros((n_samples,))
        X_rec = self._transform(X)
        scores = self._rec_error(X, X_rec)

        return scores
