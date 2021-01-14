import numpy as np
import scipy.io as scio
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split


def evaluate_print(y, y_pred, verbose=1):
    """Evaluate the AUC and Precision given y and the prediction results."""

    assert y.shape[0] == y_pred.shape[0]

    n_samples, n_outliers = y.shape[0], int(np.sum(y))
    roc = np.round(roc_auc_score(y, y_pred), decimals=4)
    indices = np.argsort(-y_pred)  # descending order
    y_pred_binary = np.zeros((n_samples,), dtype=np.uint8)
    y_pred_binary[indices[:n_outliers]] = 1

    precision = np.round(precision_score(y, y_pred_binary), decimals=4)

    if verbose > 0:
        print("ROC:{}, Precision @ rank n:{}".format(roc, precision))

    return roc, precision


def train_test_split_from_mat(data_dir, test_size=0.4, random_state=None):
    """Load and split mat data from `data_dir` in the one-class setting."""

    # Load data
    data = scio.loadmat(data_dir)
    X, y = data["X"], data["y"]
    inlier_X, inlier_y = X[y.reshape(-1) == 0, :], y[y.reshape(-1) == 0, :]
    outlier_X, outlier_y = X[y.reshape(-1) == 1, :], y[y.reshape(-1) == 1, :]

    # Split data:
    #     Train: 60 % of inliers
    #     Test: 40 % of inliers + All outliers
    X_train, tmp_X, y_train, tmp_y = train_test_split(
        inlier_X, inlier_y, test_size=test_size, random_state=random_state
    )

    X_test = np.vstack((tmp_X, outlier_X))
    y_test = np.vstack((tmp_y, outlier_y))

    return X_train, y_train, X_test, y_test
