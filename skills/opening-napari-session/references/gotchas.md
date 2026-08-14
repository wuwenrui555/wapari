# Traps

Shared by every skill in this repository. Each of these fails silently or presents as a different problem.

## Screenshots come back black on macOS

`napari.Viewer(show=False)` never gets a valid OpenGL context, and `viewer.screenshot()` then returns an all-zero image **without raising**. `QT_QPA_PLATFORM=offscreen` does not help either; Qt reports that the platform plugin cannot create an OpenGL context.

The window has to be shown, and the canvas has to have drawn at least once:

```python
viewer = napari.Viewer()          # not show=False
from qtpy.QtWidgets import QApplication
for _ in range(12):
    QApplication.processEvents()
shot = viewer.screenshot(canvas_only=True)
```

A black screenshot is therefore evidence about the canvas, not about the data. Check `layer.data` and the contrast limits before concluding an image is empty.

## Shapes and Points layers have no undo

Cmd+Z works inside a Labels layer, where paint, fill and erase keep a history. It does nothing in a Shapes or Points layer: a deleted shape is gone, and a mis-dragged vertex cannot be reverted. This is a known gap dating to 2019, not a configuration problem.

Consequences worth passing to the user: export annotations to geojson often enough that a mistake costs one shape rather than an afternoon, and prefer a Labels layer when the annotation is really a painted region, since that buys undo back.

## Delete does nothing to a polygon drawn in a Labels layer

A Labels layer has a polygon tool, and it looks like the one in a Shapes layer, but it commits **pixels**. There is no shape object afterwards, so selecting it and pressing Delete does nothing at all — Delete only removes shapes from a Shapes or Points layer.

Undo the pixels instead. Cmd+Z works here, unlike in a Shapes layer: paint, fill and polygon operations on a Labels layer keep a history. The other ways are the fill tool with the label set to 0, or, for a whole label at once:

```python
layer = viewer.layers["Labels"]
layer.data[layer.data == label] = 0
layer.refresh()
```

Shapes gives you deletable objects but no undo; Labels gives you undo but nothing to delete.

## Clicks land somewhere else: the layer is in transform mode

Transform is one of the tools in the layer's toolbar, and in it a click-drag moves the whole layer instead of painting or drawing. The only sign is a blue box with square handles around the layer on the canvas — easy to miss on a dark image, and easy to enter by mis-keying.

Press `2` for the brush, `3` for the polygon tool, `1` for the eraser, `4` for the fill bucket. The status bar lists these.

If the layer really was dragged, do not rebuild it — that would throw away everything painted so far. Put it back where the image is, which leaves the data untouched:

```python
labels, image = viewer.layers["Labels"], viewer.layers["DAPI"]
labels.translate, labels.scale, labels.rotate = image.translate, image.scale, 0
```

Before blaming geometry, check the mode: `layer.mode`. A misaligned Labels layer is a real and separate problem (napari/napari#5512, where a new Labels layer does not inherit the image's scale), but it is not the common cause.

## The Labels polygon preview draws far from the cursor

On a whole-slide Labels layer the polygon tool's preview, the in-progress outline with its white vertex handles, appears in a different part of the canvas from the clicks. The committed pixels land correctly, so the annotation is right and only the drawing is blind.

The cause is the GPU texture limit. A layer larger than `GL_MAX_TEXTURE_SIZE` on any axis is downsampled before it is uploaded, and napari says so:

```text
UserWarning: data shape (24480, 11520) exceeds GL_MAX_TEXTURE_SIZE 16384
in at least one axis and will be downsampled.
```

The polygon overlay does not follow that downsampling. Measured on a 48960 x 23040 slide against a 16384 limit:

| labels array | over the limit | preview |
| --- | --- | --- |
| 48960 x 23040 | yes, by 3x | wrong |
| 24480 x 11520 | yes | wrong |
| 16320 x 7680 | no, by 64 px | correct |
| 12240 x 5760 | no | correct |

Ruled out along the way: the layer's transform, `corner_pixels`, zoom level, the number of layers, whether the image underneath is multiscale, and float32 vertex precision.

napari's own new-labels button walks straight into this, because it sizes the layer to the whole scene at the finest step any layer uses. Fix the button rather than avoiding it:

```python
from wapari.annotate import patch_new_labels, gpu_texture_limit
patch_new_labels(viewer, limit=gpu_texture_limit())
```

It then builds the finest layer that still fits, keeping it aligned through `scale`. On this slide that is 16320 x 7680: 3-pixel precision, fine for regions and wrong for cell-level work, and a 1.1 GB array becomes 125 MB.

## Select versus direct select

In a Shapes layer, `S` (select) shows a transform box whose corner handles scale the whole shape, while `D` (direct select) edits a single vertex. "I dragged a corner and only that corner moved" means mode `D` was active; the rectangle is now an arbitrary quadrilateral.

Nothing recovers it (see undo, above), but the four vertices can be squared off from the console:

```python
import numpy as np  # napari's console does not preload it

layer = viewer.layers["Shapes"]
d = layer.data[i]
(y0, x0), (y1, x1) = d.min(0), d.max(0)
data = list(layer.data)
data[i] = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
layer.data = data
```

## A pyramid level's zarr store is not uniformly an array

`zarr.open(series.aszarr())` returns a **Group** keyed `"0"`, `"1"`, … for a multi-level series, but a bare **Array** for a single-level one. Open the series store once and index it by level rather than calling `aszarr()` per level, and handle both shapes.

## Writing a synthetic qptiff for tests needs two non-obvious flags

tifffile only groups pages into one `Baseline` CYX series with pyramid levels when the Software tag starts with `PerkinElmer-QPI` — that string is the whole of its `is_qpi` check — and when `metadata=None` is passed to every `write()`. Without the second, tifffile stamps its own shape JSON, `is_shaped` wins the series-detection race, and each page becomes a separate series with no pyramid.

Real qptiff pyramids are **flat sequential pages** — every channel at full resolution, then a thumbnail, then every channel at each sub-resolution — not subifds. A fixture built with subifds does not exercise the same code path.

## AF_UNIX socket paths are short

The operating system caps them at 104 bytes on macOS and 108 on Linux, and reports only `OSError: AF_UNIX path too long`. A path inside a deep project directory, or a pytest `tmp_path`, can exceed it.

Both bridge scripts check the length first and explain the problem, so this bites elsewhere: any other tool that opens a Unix socket under a long path fails with the bare OSError and no hint about which path was at fault.
