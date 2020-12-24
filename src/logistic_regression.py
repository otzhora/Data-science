import numpy as np
from tqdm.notebook import tqdm

from src.utils import sigmoid


def gradient_descent(
    X: np.array,
    y: np.array,
    lr: float = 0.1,
    C: float = 0,
    epochs: int = 10000,
    eps: float = 1e-5,
    initial_approximation: np.array = None,
) -> np.array:
    """
    Gradient descent for logistic regression. I assume we had two classes
    :param X: features matrix
    :param y: target value
    :param lr: learning rate
    :param C: L2 regularization strength
    :param epochs: upper bound on steps
    :param eps: stop criteria
    :param initial_approximation: initial approximation for w
    :return: weight for logistic regression and deltas.
    """
    if initial_approximation is not None:
        w = initial_approximation
    else:
        w = np.zeros(X.shape[0])

    for _ in tqdm(range(epochs)):
        p = -np.sum(X * w[:, None], axis=0) * y
        in_brackets = 1 - sigmoid(p)
        s = X * in_brackets[None] * y
        new_w = w + lr * np.mean(s, axis=1) - lr * C * w

        if np.linalg.norm(new_w - w) < eps:
            break

        w = new_w

    return w


def logistic_regression(X: np.array, w: np.array) -> np.array:
    """
    Calculate class probabilities on data X with weight w. I
    assume we had two classes
    :params X: features matrix
    :params w: weight vector
    :return: probabilities for classes.
    """
    return sigmoid(-X.T @ w)
