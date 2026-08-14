"""Labels layers that are safe to draw on.

A layer larger than the GPU's maximum texture size is downsampled by
napari before it is uploaded, with a warning, and the Labels polygon
tool's preview does not follow that downsampling. The outline then
appears somewhere other than the cursor while the committed pixels land
correctly, which makes the tool unusable by eye.

Measured on a 48960 x 23040 slide against a 16384 limit:

===================  =================  =======
array                over the limit     preview
===================  =================  =======
48960 x 23040        yes, by 3x         wrong
24480 x 11520        yes                wrong
16320 x 7680         no, by 64 px       correct
12240 x 5760         no                 correct
===================  =================  =======

So a Labels layer is created at the finest resolution that still fits,
and `scale` keeps it aligned to the image it annotates.
"""

import numpy as np

#: The limit on every GPU seen so far. Pass :func:`gpu_texture_limit`
#: when the device might report less.
DEFAULT_MAX_TEXTURE_SIZE = 16384


def texture_safe_factor(shape, limit: int = DEFAULT_MAX_TEXTURE_SIZE) -> int:
    """Return the smallest whole factor that brings ``shape`` within ``limit``.

    Smallest, because every step costs annotation precision: a whole
    slide needs 3, and rounding that up to 4 throws away detail for
    nothing.
    """
    factor = 1
    while any(-(-length // factor) > limit for length in shape):
        factor += 1
    return factor


def gpu_texture_limit() -> int:
    """Read the real limit from the GPU.

    Requires a live OpenGL context. Calling it without one takes the
    process down rather than raising, so this is never called on the
    library's own path. A caller that already has a viewer on screen can
    pass the result to :func:`add_labels_for` as ``limit``.
    """
    from vispy.gloo import gl

    return int(gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE))


def patch_new_labels(viewer, limit: int | None = None):
    """Make the viewer's new-labels button produce a usable layer.

    napari sizes a new Labels layer to the whole scene at the finest step
    any layer uses, which on a whole slide is several times the texture
    limit. The button then hands back a layer whose polygon preview draws
    in the wrong place. This keeps the button and fixes what it builds.

    Returns a callable that puts napari's own version back.
    """
    if limit is None:
        limit = DEFAULT_MAX_TEXTURE_SIZE
    original = viewer._new_labels

    def new_labels() -> None:
        extent = viewer.layers.extent
        step = np.asarray(extent.step)
        corner = np.asarray(extent.world[0])
        span = np.asarray(extent.world[1]) - corner
        shape = np.round(span / step).astype(int) + 1
        factor = texture_safe_factor(shape, limit)
        reduced = tuple(-(-int(length) // factor) for length in shape)
        viewer.add_labels(
            np.zeros(reduced, dtype="uint8"),
            scale=tuple(step * factor),
            translate=tuple(corner),
        )

    object.__setattr__(viewer, "_new_labels", new_labels)
    rewired = _rewire_new_labels_button(viewer, new_labels, original)

    def restore() -> None:
        object.__setattr__(viewer, "_new_labels", original)
        rewired(original, new_labels)

    return restore


def _rewire_new_labels_button(viewer, new_labels, original):
    """Point the toolbar button at ``new_labels``.

    Replacing the method is not enough: the button connects to the bound
    method when it is built, so it keeps calling whatever was there at
    that moment. Returns a callable that swaps the connection back.
    """

    def swap(connect, disconnect):
        try:
            from qtpy.QtWidgets import QPushButton

            window = viewer.window._qt_window
        except (AttributeError, ImportError):
            return  # a ViewerModel has no window, which is fine
        for button in window.findChildren(QPushButton):
            if "new labels layer" in button.toolTip().lower():
                try:
                    button.clicked.disconnect(disconnect)
                except (TypeError, RuntimeError):
                    button.clicked.disconnect()
                button.clicked.connect(lambda *_: connect())

    swap(new_labels, original)
    return swap


def add_labels_for(
    viewer,
    image,
    *,
    name: str = "annotations",
    factor: int | None = None,
    shape: tuple[int, int] | None = None,
    data: np.ndarray | None = None,
    dtype: str = "uint8",
    limit: int | None = None,
):
    """Add a Labels layer aligned to an image and safe to draw on.

    Parameters
    ----------
    viewer : napari.Viewer or ViewerModel
        The viewer to add to.
    image : str or napari.layers.Image
        The image to annotate, by name or by layer. Its full-resolution
        level defines the coordinate system.
    name : str
        Layer name. An existing layer of this name is replaced, so a
        session does not accumulate annotations-1, annotations-2, …
    factor : int, optional
        Downsampling factor. By default the finest one that keeps every
        axis within the GPU texture limit.
    shape : tuple, optional
        Full-resolution shape to cover. Defaults to the image's own.
    data : numpy.ndarray, optional
        Existing labels to carry over. Must match the layer's shape.
    dtype : str
        Label dtype. uint8 holds 255 regions in a quarter of the memory
        napari's own default would use.
    limit : int, optional
        Maximum texture size, defaulting to the common 16384. Pass
        :func:`gpu_texture_limit` when a viewer is already on screen and
        the device might report less.

    Returns
    -------
    napari.layers.Labels
        The new layer, selected and in polygon mode.
    """
    if isinstance(image, str):
        try:
            image = viewer.layers[image]
        except KeyError:
            raise KeyError(
                f"no layer named {image!r}; the viewer holds "
                f"{', '.join(layer.name for layer in viewer.layers)}"
            ) from None

    if shape is None:
        first = image.data[0] if isinstance(image.data, list) else image.data
        shape = tuple(first.shape[-2:])
    if limit is None:
        limit = DEFAULT_MAX_TEXTURE_SIZE
    if factor is None:
        factor = texture_safe_factor(shape, limit)

    # Round up, so the layer covers the last partial row and column
    # rather than leaving a sliver of the image un-annotatable.
    reduced = tuple(-(-length // factor) for length in shape)
    if data is None:
        data = np.zeros(reduced, dtype=dtype)
    elif tuple(data.shape) != reduced:
        raise ValueError(
            f"data shape {tuple(data.shape)} does not match the layer shape "
            f"{reduced} implied by factor {factor}"
        )

    if name in viewer.layers:
        viewer.layers.remove(name)
    labels = viewer.add_labels(
        data,
        name=name,
        scale=tuple(np.asarray(image.scale)[-2:] * factor),
        translate=tuple(np.asarray(image.translate)[-2:]),
    )
    labels.mode = "polygon"
    viewer.layers.selection.clear()
    viewer.layers.selection.add(labels)
    return labels
