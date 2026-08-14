"""Make showing a layer select it.

napari keeps visibility and selection independent, so clicking a layer's
eye icon shows it without selecting it. Adjusting its contrast or its
colormap then takes a second click, and worse, an adjustment made before
that second click silently lands on whatever was selected before. QuPath
does not work this way, and the mismatch is a standing napari proposal
(napari/napari#7532).

Hiding a layer leaves the selection alone.
"""

import weakref

# Keyed weakly by viewer: an entry keyed by id() would outlive its
# viewer and then answer for whatever object reused the address.
_CONNECTED: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _select_only(viewer, layer) -> None:
    """Make ``layer`` the whole selection.

    Not an addition to it: contrast and colormap edits apply to every
    selected layer, so a leftover member would change a layer the user
    never touched.
    """
    viewer.layers.selection.clear()
    viewer.layers.selection.add(layer)
    viewer.layers.selection.active = layer


def select_on_show(viewer):
    """Select a layer whenever it is made visible.

    Parameters
    ----------
    viewer : napari.Viewer or napari.components.ViewerModel
        The viewer to change. Layers added later are covered too.

    Returns
    -------
    callable
        Call it to restore napari's own behaviour.
    """
    if viewer in _CONNECTED:
        return _make_disconnect(viewer)

    connected: list = []
    _CONNECTED[viewer] = connected

    def on_visible(event) -> None:
        layer = event.source
        if layer.visible and layer in viewer.layers:
            _select_only(viewer, layer)

    def watch(layer) -> None:
        layer.events.visible.connect(on_visible)
        connected.append((layer, on_visible))

    for layer in viewer.layers:
        watch(layer)

    # Adding a layer emits no visibility change, so a new layer is only
    # watched from here on; it does not steal the selection on arrival.
    def on_inserted(event) -> None:
        watch(event.value)

    viewer.layers.events.inserted.connect(on_inserted)
    connected.append((viewer.layers, on_inserted))

    return _make_disconnect(viewer)


def _make_disconnect(viewer):
    def disconnect() -> None:
        for source, callback in _CONNECTED.pop(viewer, []):
            events = getattr(source, "events", source)
            if hasattr(events, "visible"):
                events.visible.disconnect(callback)
            else:
                events.inserted.disconnect(callback)

    return disconnect
