RecForest
=========

This is the official implementation of RecForest for anomaly detection, proposed in the paper "Reconstruction-based Anomaly Detection with Completely Random Forest," SIAM International Conference on Data Mining (SDM), 2021. It is highly optimized and provides Scikit-Learn like APIs.

Installation
------------

RecForest is available at `PyPI <https://pypi.org/>`__:

.. code:: bash

    $ pip install recforest

Build from Source
*****************

To use the latest version of RecForest, you will need to install the package from source:

.. code:: bash

    $ git clone https://github.com/xuyxu/RecForest.git
    $ cd RecForest
    $ python setup.py install

Notice that a C compiler is required to compile the pyx files (e.g., GCC on Linux, and MSVC on Windows). Please refer to `Cython Installation <https://cython.readthedocs.io/en/latest/src/quickstart/install.html>`__ for details.

Example 
-------

The code snippet below presents the example on how to use RecForest for anomaly detection. Scripts on reproducing experiment results in the paper are available in the directory ``examples``.

.. code:: python

    from recforest import RecForest

    model = RecForest()
    model.fit(X_train)
    anomaly_score = model.predict(X_test)

Documentation
-------------

RecForest only has two hyper-parameters: ``n_estimators`` and ``max_depth``. Docstrings on the input parameters are listed below.

* ``n_estimators``: Specify the number of decision trees in Recforest;
* ``max_depth``: Specify the maximum depth of decision trees in Recforest;
* ``n_jobs``: Specify the number of processors for joblib parallelization. ``-1`` means using all processors;
* ``random_state``: Specify the random state for reproducibility.

RecForest has three public methods. Docstrings on these methods are listed below. Notice that for all methods, the accepted data format of input ``X`` is numpy array of the shape (n_samples, n_features).

* ``fit(X)``: Fit a RecForest using the input data X;
* ``apply(X)``: Return the leaf node ID of input data X in each decision tree;
* ``predict(X)``: Return the anomaly score on the input data X.

Package Dependencies
********************

* numpy >= 1.13.3
* scipy >= 0.19.1
* joblib >= 0.12
* cython >= 0.28.5
* scikit-learn >= 0.22

A Python environment installed from `conda <https://www.anaconda.com/>`__ is highly recommended. In this case, there is no need to install any package listed above.
