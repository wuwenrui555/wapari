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
BLOCK = 512


class BlockCache:
    """Serves point reads out of the last blocks fetched.

    A chunked store charges the same for one pixel as for the chunk holding it:
    on a 2048 x 2048 OME-Zarr chunk a single point costs about 4 ms, and so does
    a 512 x 512 block. Paying once per block instead of once per pixel is what
    keeps a cursor readout responsive over a pyramid.

    The cached block keeps a reference to the array it came from, so an id is
    never reused for a different array while its entry is alive.
    """

    def __init__(self, block: int = BLOCK, keep: int = 16):
        self.block = block
        self._keep = keep
        self._blocks: dict[tuple, tuple[Any, np.ndarray]] = {}

    def read(self, plane, index: np.ndarray) -> Any:
        corner = (index // self.block) * self.block
        key = (id(plane), *corner)
        entry = self._blocks.get(key)
        if entry is None:
            window = tuple(slice(c, c + self.block) for c in corner)
            if len(self._blocks) >= self._keep:
                self._blocks.pop(next(iter(self._blocks)))
            entry = (plane, np.asarray(plane[window]))
            self._blocks[key] = entry
        return entry[1][tuple(index - corner)].item()


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


def _value_at(layer, position, cache: BlockCache | None) -> Any:
    """The layer's own value at a world position, at full resolution."""
    index = np.round(np.asarray(layer.world_to_data(position))).astype(int)
    plane = layer.data[0] if layer.multiscale else layer.data
    shape = np.asarray(plane.shape)
    if index.size != shape.size or np.any(index < 0) or np.any(index >= shape):
        return None
    if cache is not None:
        return cache.read(plane, index)
    return np.asarray(plane[tuple(index)]).item()


def read_values(viewer, position, cache: BlockCache | None = None) -> ValueTable:
    """Read every visible image layer at ``position``, given in world coordinates.

    Parameters
    ----------
    viewer : napari.Viewer or napari.components.ViewerModel
        The viewer to read from. Only visible image layers take part.
    position : sequence of float
        A world-coordinate position, such as ``event.position`` from a mouse
        callback or ``viewer.cursor.position``.
    cache : BlockCache, optional
        Reuse one across calls to avoid re-reading a chunked store for every
        pixel. Without it each read goes straight to the array.
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
        read[(row, column)] = _value_at(layer, position, cache)

    values = {(r, c): read.get((r, c)) for r in rows for c in columns}
    return ValueTable(tuple(rows), tuple(columns), values)
