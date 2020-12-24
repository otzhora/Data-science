import numpy as np


def sigmoid(p: float):
    """
    Calculate sigmoid function
    """
    return 1. / (1. + np.exp(p))
