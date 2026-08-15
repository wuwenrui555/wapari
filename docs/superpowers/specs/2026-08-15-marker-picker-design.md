# Dragging a slide in: the marker picker

## Purpose

Dragging an OME-Zarr onto a napari window should put the slide on screen without putting forty-five layers on screen. The reader parses the store, asks which markers are wanted, and adds only those.

This also removes a dependency that answers the same question worse. `napari-ome-zarr` 0.10.0 splits a `(c, y, x)` image into one layer per channel and hands each of them **level 0 only**: measured on the 45-channel 48960 x 23040 slide, whose store holds six levels, every layer came back single-scale at full resolution. An NGFF 0.4 store behaves the same way, so it is a property of the plugin's channel-splitting path rather than of the version. `wapari.image.open_image` already returns all six levels, so wapari reads its own form better than the plugin does.

## Non-goals

The picker appears when a person drags a file in, and not otherwise. It is a one-shot dialog, not a panel that stays around: a marker chosen later is chosen by asking the agent, which is what the agent is for.

It also does no grouping. Roles and purposes live in `wapari.markers` and belong to the conversation, where "put up what I need to segment this" is one sentence. The dialog is a flat alphabetical list with a filter.

## Components

| Module | Holds | Depends on |
| --- | --- | --- |
| `wapari/picker.py` | `MarkerSelection`, the state machine, and `MarkerDialog`, a thin Qt shell over it | `re`, qtpy |
| `wapari/napari_reader.py` | the npe2 reader: recognise the path, parse it, decide whether to ask | `image`, `picker`, `display` |
| `wapari/display.py` | `add_channels(viewer, path, names)`, the non-interactive entry point | `image`, `color` |
| `napari.yaml` | plugin manifest, referenced from an entry point in `pyproject.toml` | |

`MarkerSelection` holds no Qt and no viewer. It takes a list of names and returns a list of names, which is what lets the part with the interesting behaviour be tested without a window. `MarkerDialog` reads and writes it and does nothing else.

## The select-all box

Three controls: a filter box, one checkbox per marker, and one box that selects everything currently shown.

The filter only filters. Selecting is always the checkboxes, and the select-all box is a checkbox over whatever the filter has left visible.

That box is both a control and a readout, which is where the care goes:

| `selected` versus `visible()` | box shows | clicking it |
| --- | --- | --- |
| every visible marker selected | checked | clears the visible ones |
| no visible marker selected | unchecked | selects the visible ones |
| some visible markers selected | partially checked | selects the visible ones |

Partial goes to selected rather than to cleared. Either direction discards the partial state, so neither is the safe one; going up matches the reading "is everything shown selected?" answered with "make it so", and the dialog's count plus its separate import button make an accidental forty-five obvious and free to undo.

The state follows the filter without anything having to keep them in step, because the box is computed rather than stored. Filter to three markers, tick select-all, then clear the filter: forty-five are now visible and three are selected, so the box draws itself partially checked. The three stay selected.

`MarkerSelection`:

```python
names        # every marker, alphabetical
pattern      # the current filter, a regular expression
selected     # set[str], the single source of truth

visible()       # names matching pattern
toggle(name)    # one marker
header_state()  # "all" | "none" | "partial", from visible() and selected
toggle_all()    # visible() all selected -> clear them, else select them
```

### Two Qt details that are not optional

The select-all box is written to by the code that also reads user clicks, which is the trap `references/gotchas.md` already records for napari layer visibility: a programmatic write emits the same signal a click does, so without care the refresh is read back as a click and the box fights the user. Connect **`clicked`**, which fires only for user interaction, rather than `stateChanged`, which fires for both. That removes the feedback loop instead of guarding against it.

The box is tri-state for display and two-state for operation. `setTristate(True)` makes a user click cycle unchecked to partial to checked, which would let a person choose "partially checked" as an instruction, and there is no such instruction. The `clicked` handler therefore ignores the state Qt moved to and applies `toggle_all()`.

### An unfinished regular expression is not an error

A filter is typed one character at a time, so it spends most of its life invalid: `[` is a syntax error on the way to `[CD]`. Compiling it raises, and neither a traceback nor an empty dialog is a reasonable response to typing.

An invalid pattern shows no matches and marks the field. Nothing is lost, since `selected` is not touched by filtering, and the next keystroke usually fixes it.

## What happens on a drag

```text
foo.ome.zarr dropped on the canvas
  napari asks each plugin: napari_get_reader(path)
  ours recognises it and returns a reader
  open_image(path)              lazy; no pixels are read
  MarkerDialog(names)           alphabetical
    cancelled -> []             nothing is added
    confirmed -> chosen names
  add_channels(viewer, path, chosen)
```

Each layer is built from what the intermediate form promises, so the reader is also the first consumer of the rendering defaults the form has been recording and nothing has been reading:

| Layer property | Source |
| --- | --- |
| data | `Image.pyramid(name)`, every level |
| `scale` | pixel size from `coordinateTransformations` |
| `contrast_limits` | `omero.channels[].window`, falling back to `Image.contrast_limits()` |
| `colormap` | `omero.channels[].color`, or `color.assign_bright_colors` when the store records white for everything, as `qptiff_to_ome_zarr` currently does |
| `blending` | `additive` |

`Image` gains the omero window and colour per channel, which is the smallest change that makes those two rows real.

### Programmatic opens do not ask

A napari reader is a function from a path to layers, and the agent's `viewer.open()` would reach the same function. Agent code runs on the GUI thread, so a modal dialog there stops the bridge until somebody clicks, and `nsend.py` gives up after its timeout while the queued code still runs.

So the reader asks only when a person dropped the file. A module-level flag turns the dialog off, and the agent uses `add_channels` with the markers it already decided on. With the dialog off the reader adds the first channel alone, which is the behaviour `README.md` already describes: one channel goes up first, so you see tissue rather than a wall of colour.

### Reading is liberal, writing is strict

The reader accepts NGFF 0.4 and 0.5. `image.py` looks for `attrs["ome"]` first and falls back to the top-level `multiscales` and `omero` of a 0.4 store, so the existing whole-slide conversion opens without being migrated first. Only 0.5 is written.

## Dependencies

`napari-ome-zarr` goes. Nothing in `src/wapari/` imports it, and what it provides is the drag-and-drop path this replaces.

`ome-zarr` moves to the dev group. Its only use is `tests/test_convert.py`, reading our output with the NGFF reference implementation. That test is the one assertion in the suite that we did not write: everything else compares our writer against our reader, both built from one reading of the spec, so a misreading passes every one of them and still produces a file nobody else can use. Measured for this change, `ome-zarr` 0.18.0 reads both 0.4 and 0.5 stores and returns the full pyramid, so the test survives the migration.

The test is strengthened while it is being moved. It currently asserts the level count and the shape of level 0; it should also assert that a third party finds the channel names and the pixel size, since those are what the form promises and what a crop lost when nobody was checking.

`README.md` explains the Python floor by `napari-ome-zarr` and `ome-zarr` needing zarr 3, which stops being the reason.

`demo.py` calls `viewer.open(..., plugin="napari-ome-zarr")` and will break. It is a scratch notebook, so what happens to that line is the author's call rather than this change's.

## Testing

| Behaviour | How |
| --- | --- |
| filter, including case and no matches | `MarkerSelection`, no Qt |
| an invalid regular expression is not an error | `MarkerSelection` |
| `header_state` over all, none and partial | `MarkerSelection` |
| partial goes to selected | its own test, since it is a decision rather than a consequence |
| filter to three, select all, clear the filter | its own test: the box turns partial and the three stay selected |
| `napari_get_reader` recognises a store and declines other paths | small store under `tmp_path` |
| a non-interactive read | level count, `scale`, and contrast limits taken from `omero` |
| a third party can read what we write | `ome-zarr`, extended as above |

`MarkerDialog` is not tested. The repository runs no GUI tests, and the shell is kept thin enough that this is a statement about how little it does rather than a gap.
