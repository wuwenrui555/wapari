"""Putting several versions of the same images side by side.

A grid of panels, one row per subject and one column per version, with a
crosshair in every cell marking the pixel under the cursor. Built for comparing
an image against a processed copy of itself -- before and after background
subtraction, two registrations, two segmentations -- where the question is
whether a difference is real or an artefact of how the two were displayed.

The layout leans on a detail of napari 0.8: each grid cell is its own viewbox
and layers are no longer translated into place, so every cell shares one world
coordinate system and one camera. A cursor position therefore means the same
pixel in every cell with no conversion, and the crosshairs line up by
construction. ``grid.stride = 2`` is what keeps an image and its crosshair in
the same cell.

Values under the cursor are not this module's business: it labels each image
with its row and column, and :func:`wapari.pixel_values.add_pixel_values`
renders whatever it finds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ARM_PX = 26.0
GAP_PX = 6.0
LINE_PX = 1.5
CROSSHAIR_SUFFIX = " +"
MAX_STATS = 1 << 20  # a million samples is plenty to read a percentile off
LOW_PERCENTILE, HIGH_PERCENTILE = 1.0, 99.0
STATS_TILES = 4  # windows per axis when a plane is too big to read whole
STATS_TILE = 256


def crosshair(
    y: float,
    x: float,
    *,
    zoom: float,
    scale: float,
    arm_px: float = ARM_PX,
    gap_px: float = GAP_PX,
    line_px: float = LINE_PX,
) -> tuple[np.ndarray, float]:
    """Four arms around a pixel, every dimension fixed in screen pixels.

    Arm length, centre gap and line width all have to be converted out of screen
    pixels and into data units. Leaving any one of them in data units means it
    grows with the zoom: a crosshair sized to look right over a whole slide turns
    into four slabs once you are down at single pixels.

    Returns the vectors and the width to draw them with.
    """
    per_px = 1.0 / (zoom * scale)
    arm, gap = arm_px * per_px, gap_px * per_px
    arms = np.array(
        [
            [[y - gap - arm, x], [arm, 0.0]],
            [[y + gap, x], [arm, 0.0]],
            [[y, x - gap - arm], [0.0, arm]],
            [[y, x + gap], [0.0, arm]],
        ]
    )
    return arms, line_px * per_px


def _full_shape(data: Any) -> tuple[int, ...]:
    """The full-resolution shape, without reading any of it."""
    level = data[0] if isinstance(data, list | tuple) else data
    return tuple(level.shape)


def _stats(data: Any) -> np.ndarray:
    """A cheap stand-in for the plane, for choosing contrast limits.

    Scanning level 0 of a slide costs hundreds of megabytes per panel to answer
    a question the coarsest level answers just as well, so take that instead and
    subsample whatever is left.
    """
    level = data[-1] if isinstance(data, list | tuple) else data
    shape = tuple(level.shape)
    if int(np.prod(shape)) <= MAX_STATS:
        return np.asarray(level)

    # A strided read would still touch every chunk, so take a few windows
    # instead: enough of the field to bracket it, few enough chunks to be quick.
    side = min(STATS_TILE, *shape)
    rows = np.linspace(0, shape[0] - side, STATS_TILES, dtype=int)
    columns = np.linspace(0, shape[1] - side, STATS_TILES, dtype=int)
    return np.concatenate(
        [
            np.asarray(level[y : y + side, x : x + side]).ravel()
            for y in rows
            for x in columns
        ]
    )


def _limits(contrast, column: str, planes: Sequence[np.ndarray]):
    if isinstance(contrast, Mapping):
        explicit = contrast.get(column)
        if explicit is not None:
            return tuple(float(v) for v in explicit)
    if contrast == "each":
        return None
    # percentiles, not the extremes: a few hot pixels would otherwise set the
    # range and leave the tissue black
    lo = min(float(np.percentile(p, LOW_PERCENTILE)) for p in planes)
    hi = max(float(np.percentile(p, HIGH_PERCENTILE)) for p in planes)
    return (lo, hi if hi > lo else lo + 1.0)


def show_comparison(
    viewer,
    sources: Mapping[str, Mapping[str, Any]],
    *,
    scale: Sequence[float] | None = None,
    contrast: str | Mapping[str, tuple[float, float]] = "row",
    crosshair_color: str = "#ff2d55",
) -> Any:
    """Lay out ``sources`` as a grid of panels with a linked crosshair.

    Parameters
    ----------
    viewer : napari.Viewer or ViewerModel
        Cleared and rebuilt; calling this again replaces the previous panel.
    sources : mapping of row to mapping of column to image
        One row per subject, one column per version. An image may be a plain
        array or a list of pyramid levels.
    scale : sequence of float, optional
        Pixel size, applied to every layer so the panels share a coordinate
        system.
    contrast : {"row", "each"} or mapping
        ``"row"`` gives every panel in a row the same limits, which is what makes
        a before-and-after comparison honest: autoscaling each panel on its own
        would lift the darker one until the difference disappeared. ``"each"``
        scales every panel separately. A mapping fixes limits per column.
    crosshair_color : str
        Colour of the crosshairs.

    Returns
    -------
    The mouse-move callback, already connected, so a caller can drive it by hand.
    """
    viewer.layers.clear()
    viewer.mouse_move_callbacks.clear()

    step = float(scale[0]) if scale is not None else 1.0
    rows = list(sources)
    columns = list(dict.fromkeys(c for row in sources.values() for c in row))
    crosses = []
    shape = None

    for row in rows:
        samples = {column: _stats(data) for column, data in sources[row].items()}
        for column in columns:
            data = sources[row].get(column)
            if data is None:
                continue
            shape = shape or _full_shape(data)
            limits = _limits(contrast, column, list(samples.values()))
            image = viewer.add_image(
                data,
                name=f"{row} {column}",
                colormap="gray",
                metadata={"pixel_values": {"row": row, "column": column}},
                **({"scale": scale} if scale is not None else {}),
                **({"contrast_limits": limits} if limits is not None else {}),
            )
            arms, width = crosshair(0.0, 0.0, zoom=viewer.camera.zoom, scale=step)
            cross = viewer.add_vectors(
                arms,
                name=f"{image.name}{CROSSHAIR_SUFFIX}",
                edge_color=crosshair_color,
                edge_width=width,
                vector_style="line",
                opacity=0.9,
                **({"scale": scale} if scale is not None else {}),
            )
            crosses.append(cross)

    viewer.grid.enabled = True
    viewer.grid.stride = 2
    viewer.grid.shape = (len(rows), len(columns))

    height, width_px = shape if shape is not None else (0, 0)
    last = {"pixel": None, "zoom": None}

    def on_move(_viewer=None, event=None) -> None:
        position = np.asarray(event.position) / step
        y, x = float(position[-2]), float(position[-1])
        if not (0 <= y < height and 0 <= x < width_px):
            return
        pixel, zoom = (int(y), int(x)), viewer.camera.zoom
        if last["pixel"] == pixel and last["zoom"] == zoom:
            return
        rewidth = zoom != last["zoom"]
        last.update(pixel=pixel, zoom=zoom)
        arms, line = crosshair(*pixel, zoom=zoom, scale=step)
        for cross in crosses:
            cross.data = arms
            if rewidth:  # only when the zoom changed, not on every move
                cross.edge_width = line

    viewer.mouse_move_callbacks.append(on_move)
    return on_move
