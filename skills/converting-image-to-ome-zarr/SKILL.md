---
name: converting-image-to-ome-zarr
description: Use when a multiplexed slide should be converted to OME-Zarr, or when deciding whether conversion is worth it. Triggers on "convert this qptiff", "转成zarr", "make an ome-zarr", "why is this so slow to read", or before work that will read the same slide many times.
---

# Converting a slide to OME-Zarr

`wapari.convert.qptiff_to_ome_zarr` copies an Akoya qptiff into OME-Zarr without ever holding a full plane in memory. It needs no viewer.

```python
from wapari.convert import qptiff_to_ome_zarr

qptiff_to_ome_zarr("<slide>.qptiff", "<slide>.ome.zarr")
```

A 53 GB, 45-channel, six-level slide took 3.6 minutes and produced 51 GB. Run it in the background and report when it finishes; do not hold a session waiting on it.

## Decide first — conversion is often not the answer

A qptiff already reads lazily, and `viewing-multiplex-image` opens one in seconds. So convert only when the copy earns its disk:

- The data will be read **repeatedly and randomly** — cropping many regions, iterating over patches, feeding a pipeline. Chunked zarr beats strip-oriented TIFF badly here.
- It will be **written back to**, or cropped to disk alongside its metadata.
- It will be **shared or moved**, where a directory of chunks copies and resumes better than one huge file.

Do **not** convert to look at a slide once, and do not convert because zarr sounds more modern. Say which of the reasons above applies before starting, and check free disk: the output is roughly the size of the source.

## What is preserved

Nothing is silently dropped, which is what makes the copy safe to treat as the working original:

- Every pyramid level, copied as-is. No downsampling is recomputed, so level *n* is bit-identical to the source's level *n*.
- Channel names, read from the qptiff's per-page `<Biomarker>` XML rather than from a filename or a panel list.
- Pixel size, from the TIFF resolution tags, recorded in the NGFF `coordinateTransformations`.
- Per-channel display windows, so napari opens it looking sensible.
- The raw per-page XML in full, under the `qptiff` attribute — exposure times and acquisition metadata stay with the pixels.
- The thumbnail, macro and label images, under `extras`.

## Always verify, and say that you did

`verify=True` is the default: after writing, the copy is compared to the source pixel by pixel, at random windows on every level, and a mismatch raises. Leave it on.

The reason is specific and worth repeating to the user: in this ecosystem a corrupted write **looks exactly like a good one**. On the wrong tifffile version an OME-TIFF writer produced a file whose size, structure, channel names and pixel size were all correct and whose pixels were wrong; the progress bar completed, nothing raised. Only comparing pixels finds that. A conversion reported as successful without verification is a claim nobody has checked.

## Reading the result

```python
from wapari.image import open_image
image = open_image("<slide>.ome.zarr")
```

The output is standard NGFF 0.4, so QuPath, Fiji and `ome-zarr` read it too. `napari-ome-zarr` needs Python 3.12 or newer to work against zarr 3 — see `../opening-napari-session/references/gotchas.md` for why an older environment fails in a confusing way.

## Other formats

Only qptiff has a converter today. OME-TIFF and CosMx exports are the obvious next inputs; the skill name already covers them, so add a function rather than a new skill when the need arises.
