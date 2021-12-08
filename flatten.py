import numpy as np

class FlattenLayer:

    def __init__(self):
        self._shape = ()

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - ND tensor with shape (n, ..., channels)
        :output - 1D tensor with shape (n, 1)
        ------------------------------------------------------------------------
        n - jumlah data
        """
        self._shape = a_prev.shape
        return np.ravel(a_prev).reshape(a_prev.shape[0], -1)
