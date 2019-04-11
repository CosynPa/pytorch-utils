import numpy as np


def cov(x, y, bias=False):
    """The covariance between x and y.

    x has shape (n, px1, ..., pxs) and y has shape (n, py1, ..., pyt). n is the sample dimension.
    The return has shape (px1, ..., pxs, py1, ..., pyt)
    """
    assert x.ndim > 0 and y.ndim > 0 and x.shape[0] == y.shape[0], \
        str.format("The first dimension of x and y should match, got {} and {}", x.shape, y.shape)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)

    x_expanded_shape = tuple(list(x.shape) + [1] * (y.ndim - 1))
    x_expanded = x.reshape(x_expanded_shape)

    y_expanded_shape = tuple([y.shape[0]] + [1] * (x.ndim - 1) + list(y.shape[1:]))
    y_expanded = y.reshape(y_expanded_shape)

    ddof = 0 if bias else 1
    return (x_expanded * y_expanded).sum(axis=0) / (x.shape[0] - ddof)


def corrcoef(x, y):
    """The correlation coefficient between x and y.

    x has shape (n, px1, ..., pxs) and y has shape (n, py1, ..., pyt). n is the sample dimension.
    The return has shape (px1, ..., pxs, py1, ..., pyt)
    """
    covariance = cov(x, y, bias=True)
    std_x = x.std(axis=0, ddof=0)
    std_y = y.std(axis=0, ddof=0)

    std_x = std_x.reshape(list(std_x.shape) + [1] * std_y.ndim)

    return covariance / std_x / std_y
