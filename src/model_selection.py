from collections.abs import Callable
from typing import Optional

import numpy as np


class CrossVal:
    """
    Class for managing indexes during cross-validation
    """

    def __init__(self, k: int):
        """
        Create class instance for managing indexes during cross-validation

        :param k: number of folds
        """
        self.k = k

    def __call__(self, X: np.array, y: Optional[np.array] = None):
        """
        Get indexes for k-fold cross-validation

        :param X: data
        :param y: labels
        :yields: indices
        """
        if y is not None:
            assert X.shape[0] == y.shape[0]
        n = X.shape[0]
        indices = np.arange(n)
        fold_size = n // self.k

        for i in range(self.k):
            current_indices = indices[i * fold_size : i * fold_size + fold_size]
            if y is None:
                yield current_indices
            else:
                yield current_indices

