"""Putting channels of a slide on screen.

The non-interactive half of the reader: it takes the markers already
decided on, so an agent or a notebook can call it without a dialog
appearing.
"""

import numpy as np

from wapari.color import assign_bright_colors
from wapari.image import Image, open_image

# What `qptiff_to_ome_zarr` writes for every channel, so it means "no
# colour was recorded" rather than "this channel is white".
_UNSET_COLOR = "FFFFFF"


def _colormap(name: str, rgb: tuple[int, int, int]):
    """A black-to-colour ramp, which is how a fluorescence channel reads."""
    from napari.utils import Colormap

    top = [value / 255 for value in rgb]
    return Colormap(colors=[[0, 0, 0, 1], [*top, 1]], name=name)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def channel_layers(source, names: list[str], *, blending: str = "additive") -> list:
    """Build one napari LayerData tuple per named channel.

    Separate from :func:`add_channels` because a napari reader plugin
    must return these tuples rather than reach for a viewer, and both
    paths have to produce the same layers.

    Parameters
    ----------
    source : str or pathlib.Path or wapari.image.Image
        The slide. A path is opened lazily and left open, since the
        layers read from it.
    names : list[str]
        Which markers, in the order they should appear.
    blending : str
        napari blending mode. Additive is what lets two markers be
        compared in one view.
    """
    image = source if isinstance(source, Image) else open_image(source)

    # Distinct colours are worked out over everything being added, so the
    # same request gives the same colours whatever order they arrive in.
    fallback = assign_bright_colors(list(names))

    scale = image.pixel_size_um or 1.0
    layers = []
    for name in names:
        index = image.channel_index(name)
        recorded = image.channel_colors[index]
        rgb = (
            _rgb(recorded)
            if recorded and recorded.upper() != _UNSET_COLOR
            else fallback[name]
        )
        limits = image.channel_windows[index] or image.contrast_limits(name)
        meta = {
            "name": name,
            "multiscale": True,
            "colormap": _colormap(name, rgb),
            "blending": blending,
            "contrast_limits": list(np.asarray(limits, dtype=float)),
            "scale": (scale, scale),
        }
        layers.append((image.pyramid(name), meta, "image"))
    return layers


def add_channels(viewer, source, names: list[str], *, blending: str = "additive"):
    """Add one layer per named channel to ``viewer``, keeping the pyramid.

    Returns
    -------
    list
        The layers added, in the order requested.
    """
    return [
        viewer.add_image(data, **meta)
        for data, meta, _ in channel_layers(source, names, blending=blending)
    ]
