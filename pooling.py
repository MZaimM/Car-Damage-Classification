from typing import Tuple
import numpy as np

class MaxPoolLayer:

    def __init__(self, pool_size: Tuple[int, int], stride: int = 2):
        """
        :param pool_size - tuple holding shape of 2D pooling window
        :param stride - stride along width and height of input volume used to
        apply pooling operation
        """
        self._pool_size = pool_size
        self._stride = stride
        self._a = None
        self._cache = {}

    def MaxPool2D(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - 4D tensor with shape(n, h_in, w_in, c)
        :output 4D tensor with shape(n, h_out, w_out, c)
        ------------------------------------------------------------------------
        n - Banyaknya data
        w_in - lebar volume input
        h_in - tinggi volume input
        c - jumlah channel volume input/output
        w_out - lebar volume output
        h_out - tinggi volume output
        """
        self._a = np.array(a_prev, copy=True)
        n, h_in, w_in, c = a_prev.shape
        h_pool, w_pool = self._pool_size
        h_out = 1 + (h_in - h_pool) // self._stride
        w_out = 1 + (w_in - w_pool) // self._stride
        output = np.zeros((n, h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                a_prev_slice = a_prev[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
        return output