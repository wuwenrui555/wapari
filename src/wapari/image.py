"""Lazy access to a multiplexed slide, whatever it is stored as.

A whole-slide panel is tens of gigabytes, so nothing here reads pixels
until an array is computed. Akoya qptiff, OME-TIFF and OME-Zarr are all
presented the same way: named channels, a pyramid of levels per channel,
and a pixel size.
"""

import pathlib
import xml.etree.ElementTree as ElementTree

import dask.array as da
import numpy as np
import tifffile
import zarr

CONTRAST_PERCENTILE = 99.5


class Image:
    """A multiplexed image opened lazily.

    Attributes
    ----------
    path : pathlib.Path
        Where the image was read from.
    channel_names : list[str]
        Marker names in panel order.
    levels : list
        One lazy (C, Y, X) array per pyramid level, largest first.
    pixel_size_um : float or None
        Size of a level 0 pixel in micrometers, if the file records it.
    """

    def __init__(
        self,
        path: pathlib.Path,
        channel_names: list[str],
        levels: list,
        pixel_size_um: float | None,
        handle=None,
    ) -> None:
        self.path = path
        self.channel_names = channel_names
        self.levels = levels
        self.pixel_size_um = pixel_size_um
        # A qptiff's levels read from an open TiffFile, so it stays open
        # for the life of the image and is closed here.
        self._handle = handle

    def close(self) -> None:
        """Release the underlying file.

        Anything already handed to a viewer stops working: a napari layer
        holding these levels reads them lazily, so it fails on the next
        pan or zoom, far from this call and with tifffile's own bare
        assertion. Close only when nothing is displaying the image.
        """
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._closed = True

    def _check_open(self) -> None:
        if getattr(self, "_closed", False):
            raise RuntimeError(
                f"{self.path.name} is closed; reopen it with open_image(). "
                "Layers created from it before the close are also dead."
            )

    def __enter__(self) -> "Image":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<Image {self.path.name}: {len(self.channel_names)} channels, "
            f"{len(self.levels)} levels, {self.level_shapes[0]}>"
        )

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """The (C, Y, X) shape of every pyramid level."""
        return [tuple(level.shape) for level in self.levels]

    def channel_index(self, channel: str | int) -> int:
        """Return the index of a channel, naming the panel if it is absent.

        An integer is a position, which is the only way to reach a
        channel in a panel that repeats a name. Otherwise an exact name
        wins, and failing that the name is matched the way
        :func:`wapari.markers.normalize` matches, so ``PDL1`` finds a
        channel the panel spells ``PD-L1`` and ``FAP`` finds
        ``FAP-biotin``.
        """
        if isinstance(channel, int | np.integer):
            position = int(channel)
            if not 0 <= position < len(self.channel_names):
                raise KeyError(
                    f"position {position} is outside {self.path.name}, which "
                    f"has {len(self.channel_names)} channels"
                )
            return position

        from wapari.markers import normalize

        exact = [i for i, name in enumerate(self.channel_names) if name == channel]
        wanted = normalize(channel)
        matches = exact or [
            i for i, name in enumerate(self.channel_names) if normalize(name) == wanted
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Multi-cycle panels repeat a nuclear stain. Taking the first
            # silently is how you display cycle 1 believing it is cycle 3.
            raise KeyError(
                f"{channel!r} appears {len(matches)} times in "
                f"{self.path.name}, at positions "
                f"{', '.join(str(i) for i in matches)}; index it by position"
            )
        raise KeyError(
            f"no channel {channel!r} in {self.path.name}; the panel is "
            f"{', '.join(self.channel_names)}"
        )

    def pyramid(self, channel: str | int, chunk: int = 2048) -> list[da.Array]:
        """Return one lazy 2D array per level for a single channel.

        The result is what napari's ``multiscale=True`` expects: it reads
        only the level and tiles currently on screen.
        """
        self._check_open()
        index = self.channel_index(channel)
        return [
            da.from_array(level, chunks=(1, chunk, chunk))[index]
            if level.ndim == 3
            else da.from_array(level, chunks=(chunk, chunk))
            for level in self.levels
        ]

    def _sample(self, channel: str | int, max_pixels: int) -> np.ndarray:
        """Read a bounded sample of a channel's smallest level."""
        self._check_open()
        index = self.channel_index(channel)
        smallest = self.levels[-1]
        height, width = smallest.shape[-2:]
        step = max(1, int(np.ceil(np.sqrt(height * width / max_pixels))))
        if smallest.ndim == 3:
            return np.asarray(smallest[index, ::step, ::step])
        return np.asarray(smallest[::step, ::step])

    def sample_size(self, channel: str | int, max_pixels: int = 4_000_000) -> int:
        """How many pixels :meth:`contrast_limits` would look at."""
        return int(self._sample(channel, max_pixels).size)

    def contrast_limits(
        self, channel: str | int, max_pixels: int = 4_000_000
    ) -> tuple[float, float]:
        """Suggest display limits for a channel.

        Measured on the smallest pyramid level, since a percentile over
        level 0 would read every pixel. A slide with no pyramid makes
        that level the full plane, so at most ``max_pixels`` of it are
        sorted; a strided read still touches every chunk it crosses, so
        this bounds memory and the sort, not I/O. The upper limit is
        nudged above the lower one for a blank channel, whose percentile
        is 0 and which napari would otherwise refuse to render.
        """
        sample = self._sample(channel, max_pixels)
        high = float(np.percentile(sample, CONTRAST_PERCENTILE))
        return 0.0, max(high, 1.0)


def _qptiff_channel_names(series: tifffile.TiffPageSeries) -> list[str]:
    """Read marker names from the per-page ``<Biomarker>`` XML.

    Only a qptiff's pages carry that XML. Other formats give pages with
    no description at all — tifffile returns bare ``TiffFrame`` objects
    beyond page 0 — or a description that is not XML, and neither is a
    reason to fail: an unnamed channel is still a usable channel.
    """
    names = []
    for i, page in enumerate(series.pages):
        name = None
        description = getattr(page, "description", "") or ""
        if description.lstrip().startswith("<"):
            try:
                root = ElementTree.fromstring(description)
            except ElementTree.ParseError:
                root = None
            if root is not None:
                for tag in ("Biomarker", "Name"):
                    found = root.find(tag)
                    if found is not None and found.text:
                        name = found.text
                        break
        names.append(name or f"channel_{i}")
    return names


def _ome_channel_names(tif: tifffile.TiffFile, count: int) -> list[str]:
    """Read channel names from OME-XML, falling back to positions."""
    names: list[str] = []
    try:
        root = ElementTree.fromstring(tif.ome_metadata or "")
    except ElementTree.ParseError:
        root = None
    if root is not None:
        names = [
            channel.attrib.get("Name") or f"channel_{i}"
            for i, channel in enumerate(root.findall(".//{*}Channel"))
        ]
    if len(names) != count:
        names = [f"channel_{i}" for i in range(count)]
    return names


def _qptiff_pixel_size_um(page: tifffile.TiffPage) -> float | None:
    try:
        num, den = page.tags["XResolution"].value
        unit = int(page.tags["ResolutionUnit"].value)
    except KeyError:
        return None
    unit_to_um = {2: 25400.0, 3: 10000.0}  # inch, centimeter
    if not num or unit not in unit_to_um:
        return None
    return unit_to_um[unit] / (num / den)


def _open_tiff(path: pathlib.Path) -> Image:
    """Open a qptiff, an OME-TIFF or a plain TIFF."""
    tif = tifffile.TiffFile(path)  # left open: the levels read from it lazily
    series = tif.series[0]
    store = zarr.open(series.aszarr(), mode="r")
    # A multi-level series maps to a group keyed by level; a single-level
    # one maps to a bare array.
    if isinstance(store, zarr.Group):
        levels = [store[str(i)] for i in range(len(series.levels))]
    else:
        levels = [store]

    count = levels[0].shape[0] if levels[0].ndim == 3 else 1
    if tif.is_ome:
        names = _ome_channel_names(tif, count)
    else:
        names = _qptiff_channel_names(series)
    if len(names) != count:
        names = [f"channel_{i}" for i in range(count)]

    return Image(
        path,
        names,
        levels,
        _qptiff_pixel_size_um(series.pages[0]),
        handle=tif,
    )


def _open_ome_zarr(path: pathlib.Path) -> Image:
    root = zarr.open_group(str(path), mode="r")
    multiscales = root.attrs["multiscales"][0]
    levels = [root[dataset["path"]] for dataset in multiscales["datasets"]]

    # `label` is optional in NGFF omero, and a writer may name only some
    # channels, so positions fill in for whatever is missing rather than
    # the panel coming back short.
    count = levels[0].shape[0]
    channels = root.attrs.get("omero", {}).get("channels", [])
    names = [
        (channels[i].get("label") if i < len(channels) else None) or f"channel_{i}"
        for i in range(count)
    ]

    pixel_size = None
    transformations = multiscales["datasets"][0].get("coordinateTransformations")
    if transformations:
        scale = transformations[0].get("scale")
        if scale:
            pixel_size = float(scale[-1])
    return Image(path, names, levels, pixel_size)


def open_image(path: str | pathlib.Path) -> Image:
    """Open a qptiff or OME-Zarr image without reading its pixels."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no image at {path}")
    if path.is_dir() or "".join(path.suffixes).endswith(".zarr"):
        return _open_ome_zarr(path)
    return _open_tiff(path)
