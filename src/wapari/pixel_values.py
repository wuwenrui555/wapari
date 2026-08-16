"""Reading the pixel under the cursor, from every visible layer at once.

napari shows one value in its status bar, for the selected layer only, so
comparing two layers at the same point means clicking between them. This reads
them all in one pass.

Two details make it more than a loop over ``layer.get_value``:

Multiscale layers are read at level 0. ``get_value`` returns whichever level is
on screen, so the number it reports changes as the user zooms, and on a pyramid
it often comes back ``None``. A pixel value that moves when you zoom is not a
measurement.

Layers may carry ``metadata["pixel_values"] = {"row": ..., "column": ...}``, and
the table then pivots on those labels instead of listing one row per layer. That
is what lets a comparison panel show markers down the side and variants across
the top, while a viewer holding unrelated layers still gets a plain list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from napari.layers import Image

DEFAULT_COLUMN = "value"


@dataclass(frozen=True)
class ValueTable:
    """What every visible layer reads at one position.

    ``values`` holds an entry for every row and column pair, ``None`` where no
    layer occupies that cell, so a caller can render the grid without checking
    for gaps.
    """

    rows: tuple[str, ...]
    columns: tuple[str, ...]
    values: dict[tuple[str, str], Any]


def _labels(layer) -> tuple[str, str]:
    label = (layer.metadata or {}).get("pixel_values") or {}
    return label.get("row", layer.name), label.get("column", DEFAULT_COLUMN)


def _value_at(layer, position) -> Any:
    """The layer's own value at a world position, at full resolution."""
    index = np.round(np.asarray(layer.world_to_data(position))).astype(int)
    plane = layer.data[0] if layer.multiscale else layer.data
    shape = np.asarray(plane.shape)
    if index.size != shape.size or np.any(index < 0) or np.any(index >= shape):
        return None
    return np.asarray(plane[tuple(index)]).item()


def read_values(viewer, position) -> ValueTable:
    """Read every visible image layer at ``position``, given in world coordinates.

    Parameters
    ----------
    viewer : napari.Viewer or napari.components.ViewerModel
        The viewer to read from. Only visible image layers take part.
    position : sequence of float
        A world-coordinate position, such as ``event.position`` from a mouse
        callback or ``viewer.cursor.position``.
    """
    rows: list[str] = []
    columns: list[str] = []
    read: dict[tuple[str, str], Any] = {}
    for layer in viewer.layers:
        if not isinstance(layer, Image) or not layer.visible:
            continue
        row, column = _labels(layer)
        if row not in rows:
            rows.append(row)
        if column not in columns:
            columns.append(column)
        read[(row, column)] = _value_at(layer, position)

    values = {(r, c): read.get((r, c)) for r in rows for c in columns}
    return ValueTable(tuple(rows), tuple(columns), values)
