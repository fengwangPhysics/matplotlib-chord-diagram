# matplotlib-chord-diagram

Plot chord diagram with [matplotlib](https://matplotlib.org).


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

An example can be found at the end of `matplotlib-chord.py`.
Here is what the diagrams look like (left and right are respectively with and
without gradient and gap, up and down are respectively sorted by domain size
or distance):

<img src="example_sort-size.png" width="390" alt="Chord diagram without gradient, sorted by size"><img src="example_gradient_sort-size.png" width="390" alt="Chord diagram without gradient, sorted by size">
<img src="example_sort-distance.png" width="390" alt="Chord diagram without gradient, sorted by distance"><img src="example_gradient_sort-distance.png" width="390" alt="Chord diagram without gradient, dorted by distance">


## Contributors

* Original author: [@fengwangPhysics](https://github.com/fengwangPhysics)
* Colormap support: [@pakitochus](https://github.com/pakitochus) (PR [#1](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/1))
* Improved arcs: [@cy1110](https://github.com/cy1110) (PR [#2](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/2))
* Tanguy Fardet:
   - improved color support (PR [#4](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/4))
   - gradients (PR [#5](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/5))
   - refactoring (PR [#6](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/6))
