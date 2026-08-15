"""Tests for wapari.display.add_channels."""

import numpy as np
import pytest
import zarr
from napari.components import ViewerModel

from wapari.display import add_channels

PANEL = ("DAPI", "CD8", "HLA1")


def _store(path, channels, shape=(3, 16, 16), levels=3):
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    datasets = []
    for i in range(levels):
        level = (shape[0], shape[1] // 2**i, shape[2] // 2**i)
        array = root.create_array(
            str(i), shape=level, dtype="uint16", dimension_names=("c", "y", "x")
        )
        array[:] = np.arange(np.prod(level), dtype=np.uint16).reshape(level)
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 0.5 * 2**i, 0.5 * 2**i]}
                ],
            }
        )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "name": "probe",
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets,
            }
        ],
        "omero": {"channels": channels},
    }
    return path


@pytest.fixture
def store(tmp_path):
    return _store(
        tmp_path / "probe.ome.zarr",
        [
            {
                "label": "DAPI",
                "color": "FFFFFF",
                "window": {"start": 0.0, "end": 900.0},
            },
            {"label": "CD8", "color": "FFFFFF", "window": {"start": 0.0, "end": 700.0}},
            {
                "label": "HLA1",
                "color": "FFFFFF",
                "window": {"start": 0.0, "end": 500.0},
            },
        ],
    )


def test_one_layer_per_requested_channel(store):
    viewer = ViewerModel()
    add_channels(viewer, store, ["DAPI", "HLA1"])
    assert [layer.name for layer in viewer.layers] == ["DAPI", "HLA1"]


def test_nothing_else_is_added(store):
    viewer = ViewerModel()
    add_channels(viewer, store, ["CD8"])
    assert len(viewer.layers) == 1


def test_the_layer_keeps_every_pyramid_level(store):
    """The whole reason this exists rather than napari-ome-zarr, which
    hands back level 0 alone."""
    viewer = ViewerModel()
    add_channels(viewer, store, ["DAPI"])
    layer = viewer.layers["DAPI"]
    assert layer.multiscale
    assert [tuple(level.shape) for level in layer.data] == [(16, 16), (8, 8), (4, 4)]


def test_the_pixel_size_becomes_the_layer_scale(store):
    viewer = ViewerModel()
    add_channels(viewer, store, ["DAPI"])
    assert tuple(viewer.layers["DAPI"].scale) == (0.5, 0.5)


def test_contrast_limits_come_from_the_recorded_window(store):
    """The file has carried these since it was written and nothing had
    read them."""
    viewer = ViewerModel()
    add_channels(viewer, store, ["DAPI", "CD8"])
    assert tuple(viewer.layers["DAPI"].contrast_limits) == (0.0, 900.0)
    assert tuple(viewer.layers["CD8"].contrast_limits) == (0.0, 700.0)


def test_a_channel_with_no_recorded_window_is_measured(tmp_path):
    path = _store(tmp_path / "nowindow.ome.zarr", [{"label": n} for n in PANEL])
    viewer = ViewerModel()
    add_channels(viewer, path, ["DAPI"])
    low, high = viewer.layers["DAPI"].contrast_limits
    assert low == 0.0
    assert high > 0.0


def test_channels_recorded_white_get_distinct_colours(store):
    """qptiff_to_ome_zarr writes white for every channel, so taking the
    file at its word would stack three white layers."""
    viewer = ViewerModel()
    add_channels(viewer, store, ["DAPI", "CD8", "HLA1"])
    tops = {tuple(layer.colormap.colors[-1][:3]) for layer in viewer.layers}
    assert len(tops) == 3


def test_a_recorded_colour_is_used(tmp_path):
    path = _store(
        tmp_path / "coloured.ome.zarr",
        [
            {"label": "DAPI", "color": "0000FF"},
            {"label": "CD8", "color": "00FF00"},
            {"label": "HLA1", "color": "FF0000"},
        ],
    )
    viewer = ViewerModel()
    add_channels(viewer, path, ["DAPI"])
    assert tuple(viewer.layers["DAPI"].colormap.colors[-1]) == (0.0, 0.0, 1.0, 1.0)


def test_layers_blend_additively(store):
    viewer = ViewerModel()
    add_channels(viewer, store, ["DAPI"])
    assert viewer.layers["DAPI"].blending == "additive"


def test_an_unknown_channel_names_the_panel(store):
    viewer = ViewerModel()
    with pytest.raises(KeyError, match="DAPI"):
        add_channels(viewer, store, ["CD4"])


def test_asking_for_nothing_adds_nothing(store):
    viewer = ViewerModel()
    add_channels(viewer, store, [])
    assert len(viewer.layers) == 0


def test_an_already_open_image_can_be_passed_instead_of_a_path(store):
    from wapari.image import open_image

    viewer = ViewerModel()
    image = open_image(store)
    add_channels(viewer, image, ["CD8"])
    assert [layer.name for layer in viewer.layers] == ["CD8"]
