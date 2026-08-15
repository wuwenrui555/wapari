"""Labels layers aligned to the image they annotate.

A layer larger than the GPU's maximum texture size is downsampled by
napari before it is uploaded, with a warning. Up to napari 0.6.x the
Labels polygon tool's preview did not follow that downsampling, so the
outline appeared somewhere other than the cursor while the committed
pixels landed correctly, which made the tool unusable by eye. That was
napari/napari#7862, fixed by napari/napari#8563 and released in 0.7.0.

Measured then, on a 48960 x 23040 slide against a 16384 limit:

===================  =================  =======
array                over the limit     preview
===================  =================  =======
48960 x 23040        yes, by 3x         wrong
24480 x 11520        yes                wrong
16320 x 7680         no, by 64 px       correct
12240 x 5760         no                 correct
===================  =================  =======

Re-measured on napari 0.8.0, the preview is correct at every one of
those sizes, so what remains is a memory trade rather than a
correctness one:

===================  =================  ========  =======
array                over the limit     memory    preview
===================  =================  ========  =======
48960 x 23040        yes, by 3x         1128 MB   correct
24480 x 11520        yes, by 2x          282 MB   correct
16320 x 7680         no, by 64 px        125 MB   correct
12240 x 5760         no                   71 MB   correct
===================  =================  ========  =======

Full resolution costs memory; a downsampled layer costs annotation
precision, 3 pixels on this slide against a median cell diameter of
about 17. :func:`add_labels_for` picks the finest layer that still fits
unless ``factor`` says otherwise, and `scale` keeps it aligned to the
image it annotates.
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
