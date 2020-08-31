"""
Create linear color gradients
"""

from matplotlib.colors import ColorConverter, LinearSegmentedColormap

import numpy as np


def linear_gradient(cstart, cend, n=10):
    '''
    Return a gradient list of `n` colors going from `cstart` to `cend`.
    '''
    s = np.array(ColorConverter.to_rgb(cstart))
    f = np.array(ColorConverter.to_rgb(cend))

    rgb_list = [s + (t / (n - 1))*(f - s) for t in range(n)]

    return rgb_list


def gradient(start, end, color1, color2, meshgrid, mask, ax, alpha):
    '''
    Create a linear gradient from `start` to `end`, which is translationally
    invarient in the orthogonal direction.
    The gradient is then cliped by the mask.
    '''
    xs, ys = start
    xe, ye = end

    Z = None

    X, Y = meshgrid

    # get the orthogonal projection of each point on the gradient start line
    if np.isclose(ye, ys):
        Z = np.clip((X - xs) / (xe - xs), 0, 1)
    else:
        Yh = ys

        if not np.isclose(xe, xs):
            norm = np.sqrt((ye-ys)*(ye-ys) / ((xe-xs)*(xe-xs)) + 1)

            Yh = ys + ((ys - ye)*(X - xs)/(xe - xs) + (Y - ys)) / norm

        # generate the image, varying from 0 to 1
        Z = np.clip((Y - Yh) / (ye - ys), 0, 1)

    # generate the colormap
    n_bin = 100

    color_list = linear_gradient(color1, color2, n_bin)

    cmap = LinearSegmentedColormap.from_list("gradient", color_list, N=n_bin)

    im = ax.imshow(Z, interpolation='bilinear', cmap=cmap,
                   origin='lower', extent=[-1, 1, -1, 1], alpha=alpha)

    im.set_clip_path(mask)
