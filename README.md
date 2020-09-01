# mpl_chord_diagram

Python module to plot chord diagrams with [matplotlib](https://matplotlib.org).


## Usage and requirements

Add the module to your ``PYTHONPATH``, then, in python script or terminal:

```python
from mpl_chord_diagram import chord_diagram
```

If you don't know how to change the ``PYTHONPATH``, use the ``example.py``
file, it provides a workaround.

The code requires ``numpy``, ``scipy`` and ``matplotlib``, you can install
them by calling

    pip install -r requirements.txt 


## Main plot function

```python
def chord_diagram(mat, names=None, width=0.1, pad=2., gap=0., chordwidth=0.7,
                  ax=None, colors=None, cmap=None, alpha=0.7,
                  use_gradient=False, show=False, **kwargs):
    """
    Plot a chord diagram.

    Parameters
    ----------
    mat : square matrix
        Flux data, mat[i, j] is the flux from i to j
    names : list of str, optional (default: no names)
        Names of the nodes that will be displayed.
    width : float, optional (default: 0.1)
        Width/thickness of the ideogram arc.
    pad : float, optional (default: 2)
        Distance between two neighboring ideogram arcs. Unit: degree.
    gap : float, optional (default: 0)
        Distance between the arc and the beginning of the cord.
    chordwidth : float, optional (default: 0.7)
        Position of the control points for the chords, controlling their shape.
    ax : matplotlib axis, optional (default: new axis)
        Matplotlib axis where the plot should be drawn.
    colors : list, optional (default: from `cmap`)
        List of user defined colors or floats.
    cmap : str or colormap object (default: viridis)
        Colormap to use.
    alpha : float in [0, 1], optional (default: 0.7)
        Opacity of the chord diagram.
    use_gradient : bool, optional (default: False)
        Whether a gradient should be use so that chord extremities have the
        same color as the arc they belong to.
    **kwargs : keyword arguments
        Available kwargs are "fontsize" and "sort" (either "size" or
        "distance").
    """
```

## Example

An example can be found in file `example.py`.
Here is what the diagrams look like (with and without gradient and gap,
up and down are sorted respectively by domain size and distance):

<img src="example_sort-size.png" width="390"
     alt="Chord diagram without gradient, sorted by size"><img
     src="example_gradient_sort-size.png" width="390"
     alt="Chord diagram without gradient, sorted by size">

<img src="example_sort-distance.png" width="390"
     alt="Chord diagram without gradient, sorted by distance"><img
     src="example_gradient_sort-distance.png" width="390"
     alt="Chord diagram without gradient, dorted by distance">


## Contributors

* Original author: [@fengwangPhysics](https://github.com/fengwangPhysics)
* Refactoring (Tanguy Fardet, PRs
  [#6](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/6) &
  [#9](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/9))
* Improved color support:
   - [@pakitochus](https://github.com/pakitochus) (PR [#1](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/1))
   - Tanguy Fardet (PRs
      [#4](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/4) for
      colors/colormaps and
      [#5](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/5) &
      [#7](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/7) for
      gradients)
* Improved arcs:
   - [@cy1110](https://github.com/cy1110) (PR [#2](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/2))
   - Tanguy Fardet (PRs
     [#6](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/6) for
     gap addition and
     [#7](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/7) for
     adaptive curvature and sorting)
