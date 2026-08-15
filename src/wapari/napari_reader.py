"""Opening a slide by dropping it on a napari window.

napari asks every reader plugin whether it recognises a path, and the
one that does returns the layers. This one recognises the intermediate
form, asks which markers are wanted, and returns those.

It replaces `napari-ome-zarr` for this repository's own files. That
plugin splits a ``(c, y, x)`` image into one layer per channel and hands
each of them level 0 alone: measured on a 45-channel 48960 x 23040 slide
whose store holds six levels, every layer came back single-scale. An
NGFF 0.4 store behaves the same way, so it is the plugin's
channel-splitting path rather than the version.
"""

import pathlib

import zarr

from wapari.display import channel_layers
from wapari.image import open_image
from wapari.picker import ask_for_markers

#: Whether a person is at the keyboard. A napari reader is reached both
#: by a drag and by ``viewer.open()``, and agent code runs on the GUI
#: thread, where a modal dialog stops the bridge until somebody clicks.
#: Turn this off and use :func:`wapari.display.add_channels` instead.
INTERACTIVE = True


def _is_intermediate_form(path: pathlib.Path) -> bool:
    """Whether the path is an OME-Zarr this reader can open."""
    if not path.is_dir():
        return False
    try:
        attrs = zarr.open_group(str(path), mode="r").attrs
    except Exception:
        return False
    return "multiscales" in attrs.get("ome", attrs)


def napari_get_reader(path):
    """napari's reader hook: return a reader for ``path``, or None."""
    # A list means several paths at once, which for slides means several
    # dialogs and no way to tell which is which. Decline rather than ask.
    if isinstance(path, list):
        return None
    if not _is_intermediate_form(pathlib.Path(path)):
        return None
    return _read


def _read(path):
    """Return LayerData tuples for the markers that were asked for."""
    image = open_image(path)
    if not INTERACTIVE:
        # One channel, so a slide opens as tissue rather than as a wall
        # of colour. Everything else is a request away.
        return channel_layers(image, image.channel_names[:1])

    chosen = ask_for_markers(
        image.channel_names, title=f"Add markers from {pathlib.Path(path).stem}"
    )
    if chosen is None:
        return []  # cancelled, which is not the same as choosing nothing
    return channel_layers(image, chosen)


def prefer_wapari(extension: str = ".zarr") -> None:
    """Make napari open ``extension`` with this reader without asking.

    napari's own builtin reader also claims ``*.zarr``, and a plugin
    cannot outrank it. With two readers matching, dropping a store on a
    window asks which one to use before it can ask which markers to put
    up, and the builtin reads the store as a bare directory of arrays:
    the pyramid levels come back in whatever order the keys iterate, and
    napari refuses them.

    The pattern ends in a separator because napari appends one to a
    directory before matching, so ``*.zarr`` never matches a store and
    ``*.zarr/`` always does. napari's own "remember this choice" writes
    the folder's absolute path instead, which settles one store rather
    than the format.

    This changes the user's napari settings, so the library never calls
    it on its own.
    """
    import os

    from napari.settings import get_settings

    settings = get_settings()
    settings.plugins.extension2reader = {
        **settings.plugins.extension2reader,
        f"*{extension}{os.sep}": "wapari",
    }
    settings.save()
