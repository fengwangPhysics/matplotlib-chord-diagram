"""
Utilities for the chord diagram.
"""

import numpy as np


def _get_normed_line(mat, i, x, start, end, is_sparse):
    if is_sparse:
        row = mat.getrow(i).todense().A1
        return (row / x[i]) * (end - start)

    return (mat[i, :] / x[i]) * (end - start)


def dist(points):
    x1, y1 = points[0]
    x2, y2 = points[1]

    return np.sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1))


def polar2xy(r, theta):
    return np.array([r*np.cos(theta), r*np.sin(theta)])
