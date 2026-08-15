"""Cell outlines that stay fast and legible while zooming.

A Labels layer can draw outlines itself, through ``contour``, but it
recomputes them on the CPU for every redraw. Measured on a 8001 x 8319
mask holding 146912 cells:

=============  =====================
contour        milliseconds per draw
=============  =====================
0 (filled)     75
1              761
2              1155
4              2158
=============  =====================

Every toggle, pan and zoom pays that again, and a thicker line costs
more. Computing the edges once and handing napari a plain image brings a
toggle back to the filled layer's cost.

Two other things fall out of doing it this way. The image is binary, so
the outlines are one colour instead of one per cell. And each pyramid
level re-finds edges on labels downsampled to that level, rather than
downsampling the edge image, which would drop the lines: one line per
cell survives at every level, so the width looks the same at any zoom.

Zooming out far enough is a different problem that no rendering trick
solves. 146912 cells across a thousand screen pixels is a hundred cells
per pixel, and outlining all of them fills 53% of the coarsest level.
Below a few pixels per cell the layer is noise, so it hides itself.
"""

import numpy as np

#: A cell narrower than this on screen carries no visible outline.
MIN_CELL_PIXELS = 3.0


def boundaries(labels: np.ndarray) -> np.ndarray:
    """Return the pixels of ``labels`` that touch a different label.

    The edge sits inside the cell, so two touching cells each keep their
    own line and the background stays empty.
    """
    labels = np.asarray(labels)
    edge = np.zeros(labels.shape, bool)
    differs = labels[:-1, :] != labels[1:, :]
    edge[:-1, :] |= differs
    edge[1:, :] |= differs
    differs = labels[:, :-1] != labels[:, 1:]
    edge[:, :-1] |= differs
    edge[:, 1:] |= differs
    return edge & (labels > 0)


def boundary_pyramid(labels: np.ndarray, min_size: int = 1024) -> list[np.ndarray]:
    """Return one binary edge image per pyramid level, coarsest last.

    Each level finds edges on labels subsampled to that level. Halving
    the edge image instead would throw away half of every line.
    """
    levels, current = [], np.asarray(labels)
    while True:
        levels.append(boundaries(current).astype(np.uint8))
        halved = current[::2, ::2]  # nearest: labels must stay labels
        if min(halved.shape) < min_size:
            return levels
        current = halved


def _median_cell_diameter(labels: np.ndarray) -> float:
    """The diameter of a cell of median area, in pixels."""
    counts = np.bincount(np.asarray(labels).ravel())[1:]
    counts = counts[counts > 0]
    if not counts.size:
        raise ValueError("this mask has no labels to outline")
    return float(2 * np.sqrt(np.median(counts) / np.pi))


def add_outlines(
    viewer,
    labels,
    *,
    name: str = "outlines",
    color: str = "yellow",
    min_size: int = 1024,
    cell_diameter: float | None = None,
    auto_hide: bool = True,
    opacity: float = 0.9,
):
    """Add cell outlines to a viewer as a multiscale image.

    Parameters
    ----------
    viewer : napari.Viewer or ViewerModel
    labels : numpy.ndarray or str or napari.layers.Labels
        The label mask, or a layer, or the name of one in the viewer.
    color : str
        One colormap for every cell. Per-cell colour belongs to the
        Labels layer, which can stay hidden underneath.
    cell_diameter : float, optional
        Median cell diameter in pixels, measured from the mask when not
        given. Sets the zoom at which the outlines stop being legible.
    auto_hide : bool
        Hide the layer once a cell is narrower than three screen pixels.

    Returns
    -------
    napari.layers.Image
        The outline layer.
    """
    source = labels
    if isinstance(labels, str):
        source = viewer.layers[labels]
    data = getattr(source, "data", source)
    scale = tuple(np.asarray(getattr(source, "scale", (1, 1)))[-2:])
    translate = tuple(np.asarray(getattr(source, "translate", (0, 0)))[-2:])

    if cell_diameter is None:
        cell_diameter = _median_cell_diameter(data)

    levels = boundary_pyramid(data, min_size=min_size)
    if name in viewer.layers:
        viewer.layers.remove(name)
    layer = viewer.add_image(
        levels,
        name=name,
        multiscale=True,
        colormap=color,
        blending="additive",
        contrast_limits=[0, 1],
        opacity=opacity,
        scale=scale,
        translate=translate,
    )
    layer.metadata["cell_diameter_px"] = cell_diameter

    if auto_hide:
        _hide_when_illegible(viewer, layer, cell_diameter * scale[0])
    return layer


def _hide_when_illegible(viewer, layer, cell_size_in_world: float) -> None:
    """Hide the layer while a cell covers too few screen pixels.

    Only hides. Writing visibility on every zoom would take the checkbox
    away from the user: turning the outlines off and then zooming used to
    turn them straight back on.
    """
    threshold = MIN_CELL_PIXELS / max(cell_size_in_world, 1e-9)
    state = {"wanted": layer.visible, "ours": False}

    def on_visible(event=None) -> None:
        if not state["ours"]:
            state["wanted"] = bool(layer.visible)

    def on_zoom(event=None) -> None:
        if layer not in viewer.layers:
            return
        # bool(), not the numpy scalar the comparison yields: Qt rejects
        # numpy.bool with a TypeError from setEnabled.
        should = state["wanted"] and bool(viewer.camera.zoom >= threshold)
        if should != layer.visible:
            state["ours"] = True
            try:
                layer.visible = should
            finally:
                state["ours"] = False

    layer.events.visible.connect(on_visible)
    viewer.camera.events.zoom.connect(on_zoom)
    on_zoom()
