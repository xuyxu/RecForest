# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3

# Author: Yi-Xuan Xu


cimport cython
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.npy_intp SIZE_t
ctypedef np.npy_int32 INT32_t
ctypedef np.npy_float64 X_DTYPE_C


cpdef _transform(const X_DTYPE_C [:, ::1] X,
                 const SIZE_t [:] X_leaves,
                 object decision_path,
                 const SIZE_t [:] feature,
                 const X_DTYPE_C [:] threshold,
                 np.ndarray[X_DTYPE_C, ndim=2] out_lower,
                 np.ndarray[X_DTYPE_C, ndim=2] out_upper):
    """
    The internal cython function used for finding the bounding boxes through
    traversing through the entire tree.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data, the data dtype should be np.float64, and the data
        layout should be strictly C-aligned.
    X_leaves : ndarray of shape (n_samples,)
        The leaf node id for each sample in input data `X`.
    decision_path : csr_matrix
        A sparse matrix that stores the traversed nodes for each sample in `X`,
        which is produced from the `decision_path` function of base tree.
    feature : ndarray of shape (n_nodes,)
        Store the feature attribute for each node in the base tree.
    threshold : ndarray of shape (n_nodes,)
        Store the splitting cut-off for each node in the base tree.

    Returns
    -------
    out_lower : ndarray of shape (n_samples, n_features)
        Lower bounds of bounding boxes for samples along all attributes.
    out_upper : ndarray of shape (n_samples, n_features)
        Upper bounds of bounding boxes for samples along all attributes.
    """

    cdef:
        SIZE_t i
        SIZE_t sample_id
        SIZE_t n_samples = X.shape[0]
        SIZE_t n_features = X.shape[1]
        list node_indexes = []

    # Collect the node_index for each sample (i.e., the traversed nodes)
    for sample_id in range(n_samples):
        node_indexes.append(decision_path.indices[
            decision_path.indptr[sample_id]:decision_path.indptr[sample_id+1]]
        )

    for i in range(n_samples):
        c_transform(X[i, :],
                    node_indexes[i],
                    i,
                    X_leaves[i],
                    feature,
                    threshold,
                    out_lower,
                    out_upper)


cdef c_transform(const X_DTYPE_C [:] X,
                 const INT32_t [:] node_index,
                 SIZE_t sample_id,
                 SIZE_t leaf_id,
                 const SIZE_t [:] feature,
                 const X_DTYPE_C [:] threshold,
                 np.ndarray[X_DTYPE_C, ndim=2] out_lower,
                 np.ndarray[X_DTYPE_C, ndim=2] out_upper):
    """The internal C function on finding the bounding boxes for a sample."""

    cdef:
        SIZE_t i
        np.npy_bool flag
        SIZE_t n_nodes = node_index.shape[0]
        SIZE_t node_id
        SIZE_t node_feature
        X_DTYPE_C node_threshold

    with nogil:

        # Traverse through all nodes
        for i in range(n_nodes):
            node_id = node_index[i]              # current node id
            node_feature = feature[node_id]      # current splitting attribute
            node_threshold = threshold[node_id]  # current splitting cut-off

            # Skip leaf nodes
            if node_id == leaf_id:
                break
    
            # Check traversed direction
            if X[node_feature] < node_threshold:
                flag = False
            else:
                flag = True
    
            # Update bounding boxes
            if not flag:
                if out_upper[sample_id, node_feature] > node_threshold:
                    out_upper[sample_id, node_feature] = node_threshold
            elif flag:
                if out_lower[sample_id, node_feature] < node_threshold:
                    out_lower[sample_id, node_feature] = node_threshold
