RecForest
=========

This is the implementation of RecForest for anomaly detection, appearing in the paper "Reconstruction-based Anomaly Detection with Completely Random Forest," SIAM International Conference on Data Mining (2021). It is highly optimized and designed to be very easy-to-use.

Installation
------------

Stable Version
**************

The stable version of RecForest is available at PyPI, and you can install it via pip:

.. code:: bash

    pip install recforest
 
Build from Source
*****************

To get the latest version of RecForest, you need to install the package from source:

.. code:: bash

    git clone https://github.com/AaronX121/RecForest.git
    cd RecForest
    python setup.py install

You will need a C compiler to compile the .pyx files, such as GCC on Linux, or MVSC on Windows. Please refer to `Cython Installation <https://cython.readthedocs.io/en/latest/src/quickstart/install.html>`__ for details.

Example 
-------

The code snippet below presents the minimal example on how to use RecForest for anomaly detection. Testing samples with larger prediction values (i.e., ``y_pred``) are more likely to be anomalies.

.. code:: python

    from recforest import RecForest
    model = RecForest()
    model.fit(X_train)
    y_pred = model.predict(X_test)

As to the related scripts on reproducing experiment results in the original paper, please refer to files in the directory: examples.

Documentation
-------------

Below are input parameters of RecForest:

* ``n_estimators``: Specify the number of decision trees in Recforest;
* ``max_depth``: Specify the maximum depth of decision trees in Recforest;
* ``n_jobs``: Specify the number of workers for parallelization;
* ``random_state``: Specify the random state for reproducibility.

RecForest has three methods:

* ``fit(X)``: Fit RecForest using the input data X;
* ``apply(X)``: Return the leaf node ID of input data X in each decision tree;
* ``predict(X)``: Predict the anomaly score on the input data X.

Package Dependencies
********************

* numpy >= 1.13.3
* scipy >= 0.19.1
* joblib >= 0.11
* cython >= 0.28.5
* scikit-learn >= 0.22