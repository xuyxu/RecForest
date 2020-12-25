import time
import numpy as np

from recforest import RecForest
from recforest.utils import evaluate_print, train_test_split_from_mat


if __name__ == "__main__":

    datadir = "ionosphere.mat"  # MODIFY THIS TO USE OTHER DATASETS

    # Parameters
    n_estimators = 100
    max_depth = None
    n_jobs = 1

    # Utils
    seeds = [000, 111, 222, 333, 444, 555, 666, 777, 888, 999]
    roc_list = []
    precision_list = []

    for i, seed in enumerate(seeds):

        # Load and split data
        X_train, y_train, X_test, y_test = train_test_split_from_mat(
            datadir=datadir, random_state=seed
        )

        # Model
        model = RecForest(n_estimators=n_estimators,
                          max_depth=max_depth,
                          n_jobs=n_jobs,
                          random_state=seed)

        tic = time.time()
        model.fit(X_train)
        toc = time.time()
        training_time = toc - tic

        tic = time.time()
        y_pred = model.predict(X_test)
        toc = time.time()
        evaluating_time = toc - tic

        # Print and record results
        msg = "Trial: {:02d} | Random State: {:03d} | "
        print(msg.format(i+1, seed), end="")
        roc, precision = evaluate_print(y_test, y_pred)
        roc_list.append(roc)
        precision_list.append(precision)

        msg = "-- Training Time: {:.3f} s | Evaluating Time: {:.3f} s"
        print(msg.format(training_time, evaluating_time))

    # Compute Avg and Std of AUC and Precision
    auc = np.array((roc_list))
    auc_mean, auc_std = np.mean(auc), np.std(auc)

    precision = np.array((precision_list))
    precision_mean, precision_std = np.mean(precision), np.std(precision)

    # Print final results
    print("\n======= Results =======")
    print("{:<15}: {:.4f}".format("AUC Avg", auc_mean))
    print("{:<15}: {:.4f}".format("AUC Std", auc_std))
    print("{:<15}: {:.4f}".format("Precision Avg", precision_mean))
    print("{:<15}: {:.4f}".format("Precision Std", precision_std))
    print("=======================\n")
