# The intermediate form

Every reader and writer in this pipeline otherwise negotiates its own conventions, and the input interface breaks the moment someone does not follow them. This document fixes one form and says where each piece of information lives, so that a consumer which has never heard of wapari can still read a slide correctly.

Everything that enters the pipeline is converted to this form first, and everything downstream reads it:

```text
qptiff, OME-TIFF, anything else  ->  the intermediate form  ->  segmentation, feature extraction, viewing
```

There is deliberately no per-channel TIFF step between the form and its consumers. The form is already chunked one channel at a time, so writing the channels out again would copy the data a second time, cost a second verification pass, and add a second chance for silent corruption. A consumer reads the channels it needs straight from the form.

## What it is

OME-Zarr, NGFF **0.5**, written as **zarr v3**. All OME metadata lives under the `ome` key of the group's attributes, which for zarr v3 means `attributes.ome` inside `zarr.json`.

NGFF 0.4 stored the same metadata as top-level `multiscales` and `omero` keys in `.zattrs`, under zarr v2. Files in that shape are readable but are not this form.

## Layout

The group holds one array per pyramid level, named `"0"`, `"1"`, … from finest to coarsest. Every array is `(c, y, x)`.

Levels are **copied from the source, never recomputed**. A vendor pyramid is what the microscope and its software produced; regenerating it would substitute our resampling for theirs and make two files that should be identical differ.

## Where each thing lives

| Information | Location |
| --- | --- |
| Channel names | `attributes.ome.omero.channels[].label`, in channel-axis order |
| Pixel size | `attributes.ome.multiscales[0].datasets[].coordinateTransformations[].scale`, per level |
| Axis order and units | `attributes.ome.multiscales[0].axes` |
| Pyramid levels | `attributes.ome.multiscales[0].datasets[].path` |
| Rendering defaults | `attributes.ome.omero.channels[].window` and `.color` and `.active` |
| Provenance | `attributes.wapari`, described below |

Channel names are the part that nothing downstream can reconstruct. A 45-channel array without them is 45 anonymous planes, and no later step can tell which marker each one is, so a writer that cannot supply them must fail rather than write the file.

## Chunking

Chunks are `(1, T, T)` with `T` defaulting to 2048: one channel, one tile.

This is a requirement of the form, not an implementation detail. It is what makes the two access patterns cost only what they ask for. Segmentation reads two or three channels of the whole region; feature extraction reads every channel one at a time. Both touch exactly the chunks they need, and neither pulls a plane it will not use.

## Provenance

`attributes.wapari` records where the data came from. It is deliberately not named after the source format: a top-level `qptiff` key would have to be joined by an `ometiff` key, and then by one more for every format added later.

```json
{
  "version": 1,
  "source": {
    "path": "/data/20251001_Xenium097.qptiff",
    "format": "qptiff",
    "metadata": {"pages": ["<Biomarker>DAPI</Biomarker>"]}
  },
  "crop": {
    "parent": "/data/20251001_Xenium097.ome.zarr",
    "label": 1,
    "bbox": {"y0": 15435, "y1": 23436, "x0": 4098, "x1": 12417}
  }
}
```

`source.metadata` holds the original vendor metadata verbatim, so that nothing in the source is lost by converting even when this form has no field for it.

`crop` is present only on a crop, and carries what would otherwise live in a loose sidecar file next to the output: which parent it came from, and which pixels of it. A crop whose origin is recorded can be traced back; one whose origin sits in a separate JSON file loses it the first time the two are moved apart.

## Reading it without wapari

The point of fixing the locations above is that reading them takes no library beyond zarr and numpy. This is the whole reader, and it is what a script running in someone else's conda environment should use:

```python
import numpy as np
import zarr


def open_form(path):
    """Return (group, channel_names, pixel_size_um, level_paths)."""
    root = zarr.open_group(path, mode="r")
    ome = root.attrs["ome"]
    multiscale = ome["multiscales"][0]
    names = [c["label"] for c in ome["omero"]["channels"]]
    levels = [d["path"] for d in multiscale["datasets"]]
    axes = [a["name"] for a in multiscale["axes"]]
    scale = multiscale["datasets"][0]["coordinateTransformations"][0]["scale"]
    return root, names, scale[axes.index("y")], levels


def tiles(root, names, name, level="0", size=2048):
    """Walk one channel in tiles, reading only the chunks each tile touches.

    Slicing the zarr array is what keeps this lazy: indexing a channel
    first, as ``root[level][c]``, reads the whole plane immediately.
    """
    channel = names.index(name)
    array = root[level]
    height, width = array.shape[-2:]
    for y0 in range(0, height, size):
        for x0 in range(0, width, size):
            yield y0, x0, np.asarray(array[channel, y0 : y0 + size, x0 : x0 + size])
```

The laziness is easy to lose. `root[level][c]` looks like a channel view and is a fully materialized plane, 532 MB for one channel of an 8001 x 8319 region at float64. Slicing all three axes at once is what keeps zarr reading only the chunks a tile covers.

Tiling is enough for the aggregations this pipeline does, because they are sums. `np.bincount` accumulated over tiles gives the same answer as one call over the whole plane, so per-cell totals never need a plane in memory:

```python
totals = np.zeros(n_labels, dtype=np.float64)
for y0, x0, tile in tiles(root, names, "HLA1"):
    labels = mask[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]]
    totals += np.bincount(labels.ravel(), weights=tile.ravel().astype(np.float64),
                          minlength=n_labels)
```

## Writing it

Any writer that produces this form compares the written pixels against the source before reporting success. Corruption in this ecosystem is silent, produces a plausible file size and well-formed metadata, and never raises. `wapari.convert.verify_conversion` is the existing implementation of that check.

## Known limitation: napari-ome-zarr discards the pyramid

`napari-ome-zarr` 0.10.0 splits a `(c, y, x)` image into one layer per channel, and each of those layers arrives with **level 0 only**. Measured on the 45-channel 48960 x 23040 slide, whose store holds six levels: every layer came back single-scale at full resolution. The same happens with an NGFF 0.4 store, so it is a property of the plugin's channel-splitting path rather than of this form or of the NGFF version.

Read through `wapari.image.open_image` instead when the pyramid matters. On the same slide it returns all six levels:

```text
(48960, 23040), (24480, 11520), (12240, 5760), (6120, 2880), (3060, 1440), (1530, 720)
```
