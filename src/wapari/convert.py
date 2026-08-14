"""Streaming conversion of pyramidal qptiff files to OME-Zarr (NGFF 0.4).

The conversion never materializes a full image plane in memory: each
pyramid level is read through tifffile's zarr interface in row bands and
written band-by-band into the output zarr arrays.
"""

import pathlib

import numpy as np
import tifffile
import zarr

from wapari.tiff import TiffZarrReader

# NGFF axes for a (C, Y, X) multiplexed image.
_AXES = [
    {"name": "c", "type": "channel"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def _pixel_size_um(page: tifffile.TiffPage) -> float | None:
    """Read the pixel size in micrometers from TIFF resolution tags."""
    try:
        num, den = page.tags["XResolution"].value
        unit = page.tags["ResolutionUnit"].value
    except KeyError:
        return None
    if num == 0:
        return None
    px_per_unit = num / den
    unit_to_um = {2: 25400.0, 3: 10000.0}  # RESUNIT.INCH, RESUNIT.CENTIMETER
    if int(unit) not in unit_to_um:
        return None
    return unit_to_um[int(unit)] / px_per_unit


def _open_levels(series: tifffile.TiffPageSeries) -> list[zarr.Array]:
    """Open every pyramid level of a series as a lazy zarr array.

    A multi-level series maps to a zarr group keyed by level index; a
    single-level series maps to a bare array.
    """
    store = zarr.open(series.aszarr(), mode="r")
    if isinstance(store, zarr.Group):
        return [store[str(i)] for i in range(len(series.levels))]
    return [store]


def _copy_level(
    src: zarr.Array, dst: zarr.Array, band_height: int, label: str, progress: bool
) -> None:
    """Copy a (C, Y, X) level in row bands so memory stays bounded."""
    n_channels, height, _ = src.shape
    for c in range(n_channels):
        for y0 in range(0, height, band_height):
            y1 = min(y0 + band_height, height)
            dst[c, y0:y1, :] = src[c, y0:y1, :]
        if progress:
            print(f"  {label}: channel {c + 1}/{n_channels} done", flush=True)


def qptiff_to_ome_zarr(
    qptiff_f: str | pathlib.Path,
    zarr_f: str | pathlib.Path,
    channel_names: list[str] | None = None,
    chunk_size: int = 2048,
    pixel_size_um: float | None = None,
    verify: bool = True,
    progress: bool = True,
    overwrite: bool = False,
) -> pathlib.Path:
    """Convert a pyramidal qptiff to OME-Zarr, preserving all metadata.

    Parameters
    ----------
    qptiff_f : str or pathlib.Path
        Path to the source qptiff file.
    zarr_f : str or pathlib.Path
        Path of the OME-Zarr directory to create (overwritten if present).
    channel_names : list[str], optional
        Channel names to record. If None, extracted from the qptiff
        ``<Biomarker>`` page metadata.
    chunk_size : int
        Output chunk edge length in pixels; chunks are (1, chunk_size,
        chunk_size). Also used as the copy band height.
    pixel_size_um : float, optional
        Pixel size in micrometers. If None, read from TIFF resolution
        tags; falls back to 1.0 if absent.
    verify : bool
        If True, run :func:`verify_conversion` after writing.
    progress : bool
        If True, print per-channel progress.

    Returns
    -------
    pathlib.Path
        The path of the written OME-Zarr directory.
    """
    qptiff_f = pathlib.Path(qptiff_f)
    zarr_f = pathlib.Path(zarr_f)

    # Writing a zarr group clears the directory first, so an existing
    # path has to be an explicit decision rather than a side effect.
    if zarr_f.exists() and not overwrite:
        raise FileExistsError(
            f"{zarr_f} already exists; pass overwrite=True to replace it, "
            "which deletes everything already in that directory"
        )

    if channel_names is None:
        channel_names = TiffZarrReader.extract_channel_names_qptiff(qptiff_f)

    with tifffile.TiffFile(qptiff_f) as tif:
        series = tif.series[0]
        page0 = series.pages[0]
        n_channels = series.levels[0].shape[0]
        if len(channel_names) != n_channels:
            raise ValueError(
                f"got {len(channel_names)} channel names for {n_channels} "
                f"channels: {', '.join(channel_names)}"
            )
        if pixel_size_um is None:
            pixel_size_um = _pixel_size_um(page0) or 1.0

        root = zarr.open_group(str(zarr_f), mode="w", zarr_format=2)

        # Copy every pyramid level as-is (no downsampling recomputed).
        datasets = []
        shape0 = series.levels[0].shape
        smallest = None
        for i, src in enumerate(_open_levels(series)):
            dst = root.create_array(
                str(i),
                shape=src.shape,
                chunks=(1, chunk_size, chunk_size),
                dtype=src.dtype,
            )
            _copy_level(src, dst, chunk_size, f"level {i}", progress)
            factor = round(shape0[-2] / src.shape[-2])
            datasets.append(
                {
                    "path": str(i),
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [
                                1.0,
                                pixel_size_um * factor,
                                pixel_size_um * factor,
                            ],
                        }
                    ],
                }
            )
            smallest = dst

        # Contrast windows from the smallest level: cheap and good enough
        # for sensible rendering defaults in napari.
        dtype_max = float(np.iinfo(series.dtype).max)
        channels_meta = []
        for c, name in enumerate(channel_names):
            plane = np.asarray(smallest[c])
            end = float(np.percentile(plane, 99.5))
            channels_meta.append(
                {
                    "label": name,
                    "active": c == 0,
                    "color": "FFFFFF",
                    "window": {
                        "start": 0.0,
                        "end": max(1.0, min(end, dtype_max)),
                        "min": 0.0,
                        "max": dtype_max,
                    },
                }
            )

        root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": qptiff_f.stem,
                "axes": _AXES,
                "datasets": datasets,
            }
        ]
        root.attrs["omero"] = {"channels": channels_meta}
        # Preserve the raw per-page XML descriptions verbatim.
        root.attrs["qptiff"] = {
            "pages": [page.description or "" for page in series.pages],
            "pixel_size_um": pixel_size_um,
        }

        # Carry over auxiliary series (thumbnail / macro / label images).
        if len(tif.series) > 1:
            extras = root.create_group("extras")
            for extra in tif.series[1:]:
                name = (extra.name or "series").lower()
                arr = extras.create_array(name, shape=extra.shape, dtype=extra.dtype)
                arr[:] = extra.asarray()
                arr.attrs["description"] = extra.pages[0].description or ""

    if verify:
        verify_conversion(qptiff_f, zarr_f)
    return zarr_f


def verify_conversion(
    qptiff_f: str | pathlib.Path,
    zarr_f: str | pathlib.Path,
    band_height: int = 2048,
) -> int:
    """Compare every pixel of the copy against the source.

    Reads both sides in row bands, so memory stays bounded whatever the
    slide's size, and raises ``RuntimeError`` on the first mismatch. A sample
    would not be enough: corruption in this ecosystem is silent and can
    sit anywhere. Returns the number of pixels compared.
    """
    root = zarr.open_group(str(zarr_f), mode="r")
    compared = 0
    with tifffile.TiffFile(qptiff_f) as tif:
        for i, src in enumerate(_open_levels(tif.series[0])):
            try:
                dst = root[str(i)]
            except KeyError:
                raise RuntimeError(f"level {i} is missing from {zarr_f}") from None
            if src.shape != dst.shape:
                raise RuntimeError(
                    f"shape mismatch at level {i}: {src.shape} vs {dst.shape}"
                )
            n_channels, height, _ = src.shape
            for c in range(n_channels):
                for y0 in range(0, height, band_height):
                    y1 = min(y0 + band_height, height)
                    band = np.asarray(src[c, y0:y1, :])
                    if not np.array_equal(band, np.asarray(dst[c, y0:y1, :])):
                        raise RuntimeError(
                            f"pixel mismatch at level {i}, channel {c}, "
                            f"rows {y0} to {y1}"
                        )
                    compared += band.size
    return compared
