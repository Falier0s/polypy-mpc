import numpy as np


def pagemtimes(a, b):
    """Perform matrix multiplication

    Args:
        a (numpy.ndarray): First matrix to multiply.
        b (numpy.ndarray): Second matrix to multiply.

    Returns:
        numpy.ndarray: The result of the matrix multiplication.
    """
    for i in range(3 - len(np.shape(a))):
        a = np.expand_dims(a, i + len(np.shape(a)))

    for i in range(3 - len(np.shape(b))):
        b = np.expand_dims(b, i + len(np.shape(b)))
    result_matrix = np.zeros((np.shape(a)[0], np.shape(b)[1], np.shape(b)[2]))
    for i in range(np.shape(a)[2]):
        result_matrix[:, :, i] = np.matmul(a[:, :, i], b[:, :, i])
    return result_matrix
