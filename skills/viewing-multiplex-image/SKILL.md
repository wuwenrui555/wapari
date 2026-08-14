---
name: viewing-multiplex-image
description: Use when the user wants to look at a multiplexed slide (CODEX / Akoya qptiff, OME-TIFF, OME-Zarr) in napari, choose which markers to display, or asks what is in a panel. Triggers on "open this qptiff", "打开这个图", "看看这张片子", "show me DAPI and CD8", "what channels are in this file", "put up the segmentation markers".
---

# Viewing a multiplexed image

A CODEX slide is tens of gigabytes and carries forty or more markers. Two things follow: **never read it eagerly**, and **never put the whole panel on screen**. Forty additive layers are a wall of colour that answers no question.

## 1. Get a viewer

Follow `opening-napari-session`. Everything below runs through that session, so `run` and `$SKILL` in the examples are the ones it defines.

## 2. Open the image and show one channel

`wapari.image.open_image` handles qptiff and OME-Zarr identically and reads no pixels. Put up the nuclear channel alone: the user sees where the tissue is within seconds, and every later decision is easier against that background.

```python
from wapari.image import open_image
from wapari import markers

image = open_image("<path>")
first = markers.nuclear_channel(image.channel_names)  # DAPI, or the first channel
viewer.add_image(
    image.pyramid(first),
    name=first,
    colormap="blue",
    blending="additive",
    multiscale=True,
    contrast_limits=image.contrast_limits(first),
)
viewer.reset_view()
```

`pyramid()` returns one lazy array per level, which is what `multiscale=True` expects: napari reads only the tiles and the level currently on screen.

## 3. Report the panel by purpose

`markers.describe(image.channel_names)` groups the panel into what each marker is for:

- **segmentation** — the nuclear channel plus every marker that draws cell boundaries. A backend takes one nuclear channel and a set of boundary markers, which is what `markers.nuclear_channel` and `markers.boundary_markers` return separately.
- **annotation** — lineage-specific markers, the ones that say what kind of cell this is.
- **other** — functional and state markers. Real, but they describe a state rather than an identity.
- **unknown** — not in the table yet.

Show the user these groups, not a flat list of forty names. A marker can appear twice: PanCK both draws epithelial boundaries and identifies epithelium, and saying so is more useful than picking one.

For anything under **unknown**, classify it yourself against those same three criteria, say which group you think it belongs to and why, and **ask the user to confirm before relying on it**. Once confirmed, offer to add it to `MARKER_ROLES` in `src/wapari/markers.py` so the next panel with that marker needs no guessing. Never silently treat a guess as settled.

## 4. Ask what they are doing, then add those markers

Ask which of these they want, and add the corresponding group:

1. **Segmentation** — DAPI plus the boundary markers.
2. **Annotation** — the lineage markers.
3. **Just looking** — leave the nuclear channel up and wait for them to name markers.

```python
chosen = markers.markers_for(image.channel_names, "annotation")
```

Add them one at a time with distinct colormaps and `blending="additive"`, and keep each one's `contrast_limits=image.contrast_limits(name)`. Beyond five or six simultaneous channels colours stop being separable, so if a group is larger, say so and offer either the first few or a subset the user names.

Useful colormaps, roughly in order of how well they separate: `blue`, `green`, `red`, `magenta`, `cyan`, `yellow`, `bop orange`, `bop purple`.

**A broadly expressed marker will drown the others.** HLA1, HLADR, VIM and CD47 stain most of the tissue, so blended additively they wash the picture out and the sparse markers disappear into it. The default limits are not wrong — the marker really is everywhere — but the display is useless. Demote it to a backdrop instead of removing it:

```python
import numpy as np
layer = viewer.layers["HLA1"]
small = np.asarray(image.levels[-1][image.channel_index("HLA1")])
layer.contrast_limits = [np.percentile(small, 60), np.percentile(small, 99.9)]
layer.opacity = 0.35
```

Raising the lower limit to around the 60th percentile cuts the pervasive background; the opacity keeps it as context. Judge this from the screenshot, not from the marker's name.

A layer can also be loaded but hidden — `layer.visible = False`. Prefer that to leaving a marker out: the user sees the whole relevant set in the layer list and toggles what they want, without waiting for anything to load.

## 5. Make showing a layer select it

Do this once per session, early, before the user starts clicking. napari keeps visibility and selection independent: clicking a layer's eye icon shows it but does not select it, so adjusting its contrast takes a second click — and an adjustment made before that click silently lands on whichever layer was selected before.

```python
from wapari.selection import select_on_show
select_on_show(viewer)
```

Showing a layer now selects it; hiding one leaves the selection alone, which is how QuPath behaves. This is a standing napari proposal (napari/napari#7532) rather than a setting, so it has to be applied per viewer. Say once that you have done it — the window the user is clicking in has just changed behaviour.

## 6. Confirm what is on screen

Screenshot and read it, as `opening-napari-session` describes. All black means the canvas did not draw, not that the channel is empty; a channel that is genuinely blank shows as a uniform field once the contrast limits are right.

## Adding markers later

The user will keep naming markers. Adding one is the same three lines, so do it directly rather than reopening the image:

```python
name = "CD68"
viewer.add_image(
    image.pyramid(name), name=name, colormap="green", blending="additive",
    multiscale=True, contrast_limits=image.contrast_limits(name),
)
```

`channel_index` takes an exact name first, then falls back to a normalised match, so `PDL1` finds a panel's `PD-L1` and `FAP` finds `FAP-biotin`. What it will not do is guess between two channels that normalise alike, which is what a multi-cycle panel's repeated DAPI looks like; it raises and asks for a position instead. A name that matches nothing raises a `KeyError` listing the panel.

## When to convert first

Reading a qptiff lazily is fast enough for looking. Convert to OME-Zarr with `converting-image-to-ome-zarr` when the same data will be read many times, cropped to disk, or shared. Do not convert just to look once.
