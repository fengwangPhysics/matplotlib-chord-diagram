# matplotlib-chord-diagram

Plot chord diagram with [matplotlib](https://matplotlib.org).


## Main plot function

```python
def chordDiagram(X, width=0.1, pad=2., chordwidth=0.7, colors=None,
                 cmap=None, alpha=0.7, ax=None):
    """
    Plot a chord diagram.

    Parameters
    ----------
    X : square matrix
        Flux data, X[i, j] is the flux from i to j
    width : float, optional (default: 0.1)
        Width/thickness of the ideogram arc.
    pad : float, optional (default: 2)
        Gap pad between two neighboring ideogram arcs. Unit: degree.
    chordwidth : float, optional (default: 0.7)
        Position of the control points for the chords, controlling their shape.
    colors : list, optional (default: from `cmap`)
        List of user defined colors or floats.
    cmap : str or colormap object (default: viridis)
        Colormap to use.
    alpha : float in [0, 1], optional (default: 0.7)
        Opacity of the chord diagram.
    ax : matplotlib axis, optional (default: new axis)
        Matplotlib axis where the plot should be drawn.
    """
```

## Example

An example can be found at the end of `matplotlib-chord.py`.
Here is what the figure looks like:
![](example.png)


## Contributors

* Original author: [@fengwangPhysics](https://github.com/fengwangPhysics)
* Colormap support: [@pakitochus](https://github.com/pakitochus) (PR [#1](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/1))
* Improved arcs: [@cy1110](https://github.com/cy1110) (PR [#2](https://github.com/Silmathoron/matplotlib-chord-diagram/pull/2))
