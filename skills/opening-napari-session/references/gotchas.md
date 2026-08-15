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

## The Labels polygon preview draws far from the cursor (napari 0.6.x and earlier)

Fixed upstream. On napari 0.7.0 and later this section is history; it is kept because the symptom is baffling, the diagnosis was expensive, and anyone pinned to an older napari still meets it.

On a whole-slide Labels layer the polygon tool's preview, the in-progress outline with its white vertex handles, appeared in a different part of the canvas from the clicks. The committed pixels landed correctly, so the annotation was right and only the drawing was blind.

The cause is the GPU texture limit. A layer larger than `GL_MAX_TEXTURE_SIZE` on any axis is downsampled before it is uploaded, and napari says so:

```text
UserWarning: data shape (24480, 11520) exceeds GL_MAX_TEXTURE_SIZE 16384
in at least one axis and will be downsampled.
```

The polygon overlay did not follow that downsampling. Measured on a 48960 x 23040 slide against a 16384 limit, under napari 0.6.5:

| labels array | over the limit | preview |
| --- | --- | --- |
| 48960 x 23040 | yes, by 3x | wrong |
| 24480 x 11520 | yes | wrong |
| 16320 x 7680 | no, by 64 px | correct |
| 12240 x 5760 | no | correct |

Ruled out along the way: the layer's transform, `corner_pixels`, zoom level, the number of layers, whether the image underneath is multiscale, and float32 vertex precision.

This is napari/napari#7862, reported by a napari maintainer in April 2025 and fixed by napari/napari#8563, released in 0.7.0. The fix applies the layer's `tile2data` inverse transform to the overlay's points, which is exactly the missing step the table above points at.

The misplacement can be measured without touching the mouse, which is worth knowing because the obvious reading is that it needs a real pointer. Set the overlay's points in data coordinates and read back what vispy was told to draw. Re-measured on napari 0.8.0, all four sizes above now come out correct:

```python
import numpy as np
from napari._vispy.overlays.labels_polygon import VispyLabelsPolygonOverlay

layer = viewer.add_labels(np.zeros((20000, 4000), np.uint8))
layer.mode = "polygon"
visual = next(
    v
    for v in viewer.window._qt_viewer.canvas._layer_overlay_to_visual[layer].values()
    if isinstance(v, VispyLabelsPolygonOverlay)
)

points = [[10000.0, 1000.0], [10000.0, 2000.0], [15000.0, 2000.0]]
layer._overlays["polygon"].points = points

downsample = np.asarray(layer._transforms["tile2data"].scale)
drawn = visual._nodes._data["a_position"][:, :2]
np.testing.assert_allclose(drawn, np.array(points)[:, ::-1] / downsample[::-1], atol=0.5)
```

What survives the fix is a memory question rather than a correctness one. napari's new-labels button still sizes the layer to the image's full resolution, so on this slide it builds a 1.1 GB array that the GPU then downsamples anyway (napari/napari#7863, still open). `wapari.annotate.add_labels_for` takes a `factor` for that trade: full resolution costs memory, and a downsampled layer costs annotation precision, 3 pixels here against a median cell diameter of about 17.

## uv keeps the old version rather than reporting a blocked upgrade

`uv lock --upgrade-package napari` printed `Resolved 204 packages` and changed nothing at all, no diff and no warning. The upgrade was blocked, and saying so is not part of what the command does.

The chain took a while to see: napari 0.8.0 requires `napari-console>=0.1.4`, which caps `ipykernel<7`, and this project declared `ipykernel>=7.1.0`. With no version of napari able to satisfy that, the resolver kept 0.6.5 and reported success.

The way to make it talk is to demand the version you want and let resolution fail:

```bash
# temporarily, in pyproject.toml: napari>=0.8.0
uv lock
```

It then names the conflict in full. The general shape is worth remembering: a silent no-op from `--upgrade-package` means blocked, not up to date, and the constraint doing the blocking is often one of your own rather than anything the package you are upgrading declares.

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

## An old environment reads OME-Zarr as nothing at all

`napari-ome-zarr` and `ome-zarr` only work against zarr 3 from versions 0.10.0 and 0.18.0, and those need Python newer than 3.11. On 3.11 a resolver quietly picks two-year-old versions that fail to import at all:

```text
ImportError: cannot import name 'FSStore' from 'zarr.storage'
```

The message names zarr, so the reflex is to change the zarr pin, which makes it worse. Check the Python version first.

## Automatic layer behaviour has to leave the checkbox working

Anything that sets `layer.visible` from an event takes that control away from the user unless it is written carefully. Turning the outlines off and then zooming turned them back on, because the zoom callback wrote visibility unconditionally.

The rule that works: an automatic behaviour may act while the user has not expressed a preference, and must not overrule one they have. Record the last explicit choice, and let the automatic rule only narrow it.

Distinguishing the two is the part that bites. A callback's own write emits the same `visible` event a click does, so without a marker around it the automatic hide is recorded as the user wanting the layer hidden, and it never comes back:

```python
state = {"wanted": layer.visible, "ours": False}

def on_visible(event=None):
    if not state["ours"]:
        state["wanted"] = bool(layer.visible)
```

## AF_UNIX socket paths are short

The operating system caps them at 104 bytes on macOS and 108 on Linux, and reports only `OSError: AF_UNIX path too long`. A path inside a deep project directory, or a pytest `tmp_path`, can exceed it.

Both bridge scripts check the length first and explain the problem, so this bites elsewhere: any other tool that opens a Unix socket under a long path fails with the bare OSError and no hint about which path was at fault.
