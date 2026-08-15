"""Tests for wapari.outlines."""

import numpy as np
import pytest
from napari.components import ViewerModel

from wapari.outlines import add_outlines, boundaries, boundary_pyramid


@pytest.fixture
def labels():
    """Four square cells in a 2x2 arrangement, with a gap between them."""
    mask = np.zeros((40, 40), np.uint32)
    mask[2:18, 2:18] = 1
    mask[2:18, 22:38] = 2
    mask[22:38, 2:18] = 3
    mask[22:38, 22:38] = 4
    return mask


def test_a_boundary_is_one_pixel_wide_inside_the_cell(labels):
    edge = boundaries(labels)
    assert edge[2, 2] and edge[2, 17]  # the cell's own rim
    assert not edge[10, 10]  # its middle
    assert not edge[0, 0]  # background is never an edge


def test_background_is_not_outlined(labels):
    """Otherwise every cell gets a double line and the gaps light up."""
    assert not boundaries(labels)[20, 20]


def test_touching_cells_both_get_an_edge():
    mask = np.zeros((10, 10), np.uint32)
    mask[:, :5] = 1
    mask[:, 5:] = 2
    edge = boundaries(mask)
    assert edge[5, 4] and edge[5, 5]


def test_the_pyramid_halves_each_level(labels):
    levels = boundary_pyramid(labels, min_size=8)
    assert [level.shape for level in levels] == [(40, 40), (20, 20), (10, 10)]


def test_every_level_is_computed_from_labels_not_from_the_edges(labels):
    """Downsampling an edge image drops the lines; downsampling the
    labels and re-finding edges keeps one line per cell at every level,
    which is what makes the width look constant while zooming."""
    levels = boundary_pyramid(labels, min_size=8)
    naive = boundaries(labels)[::2, ::2]
    assert levels[1].sum() > naive.sum()


def test_the_pyramid_is_binary(labels):
    """A single colour, not 146912 random ones."""
    for level in boundary_pyramid(labels, min_size=8):
        assert set(np.unique(level)) <= {0, 1}
        assert level.dtype == np.uint8


def test_outlines_are_added_as_one_multiscale_layer(labels):
    viewer = ViewerModel()
    layer = add_outlines(viewer, labels, min_size=8)
    assert layer.multiscale
    assert len(layer.data) == 3
    assert layer.name == "outlines"


def test_the_source_layer_can_be_given_by_name(labels):
    viewer = ViewerModel()
    viewer.add_labels(labels, name="cells")
    layer = add_outlines(viewer, "cells", min_size=8)
    assert tuple(layer.scale) == tuple(viewer.layers["cells"].scale)


def test_outlines_inherit_the_labels_placement(labels):
    viewer = ViewerModel()
    viewer.add_labels(labels, name="cells", scale=(3, 3), translate=(5, 7))
    layer = add_outlines(viewer, "cells", min_size=8)
    assert tuple(layer.scale) == (3.0, 3.0)
    assert tuple(layer.translate) == (5.0, 7.0)


def test_outlines_hide_when_a_cell_is_smaller_than_a_few_pixels(labels):
    """Zoomed out, 146912 outlines cover half the pixels and read as a
    solid block. Below legibility the layer is worse than nothing."""
    viewer = ViewerModel()
    add_outlines(viewer, labels, min_size=8, cell_diameter=16)
    viewer.camera.zoom = 1.0
    assert viewer.layers["outlines"].visible
    viewer.camera.zoom = 0.01
    assert not viewer.layers["outlines"].visible


def test_outlines_come_back_on_zoom_in(labels):
    viewer = ViewerModel()
    add_outlines(viewer, labels, min_size=8, cell_diameter=16)
    viewer.camera.zoom = 0.01
    viewer.camera.zoom = 2.0
    assert viewer.layers["outlines"].visible


def test_auto_hiding_can_be_turned_off(labels):
    viewer = ViewerModel()
    add_outlines(viewer, labels, min_size=8, cell_diameter=16, auto_hide=False)
    viewer.camera.zoom = 0.01
    assert viewer.layers["outlines"].visible


def test_the_cell_diameter_is_measured_when_not_given(labels):
    """The threshold depends on how big the cells actually are, and the
    mask knows: 16x16 squares here."""
    viewer = ViewerModel()
    layer = add_outlines(viewer, labels, min_size=8)
    assert 12 <= layer.metadata["cell_diameter_px"] <= 20


def test_a_second_call_replaces_the_layer(labels):
    viewer = ViewerModel()
    add_outlines(viewer, labels, min_size=8)
    add_outlines(viewer, labels, min_size=8)
    assert [layer.name for layer in viewer.layers].count("outlines") == 1


def test_an_empty_mask_is_refused():
    viewer = ViewerModel()
    with pytest.raises(ValueError, match="no labels"):
        add_outlines(viewer, np.zeros((10, 10), np.uint32))


def test_visibility_is_a_plain_bool(labels):
    """camera.zoom is a numpy scalar, so the comparison yields numpy.bool,
    and Qt refuses it: 'setEnabled(): argument 1 has unexpected type'.
    A headless ViewerModel accepts it, so only a real window catches this."""
    viewer = ViewerModel()
    add_outlines(viewer, labels, min_size=8, cell_diameter=16)
    viewer.camera.zoom = np.float64(0.01)
    assert type(viewer.layers["outlines"].visible) is bool
    viewer.camera.zoom = np.float64(2.0)
    assert type(viewer.layers["outlines"].visible) is bool


def test_hiding_the_layer_by_hand_survives_a_zoom(labels):
    """Auto-hide may hide, never show. Turning outlines off and then
    zooming used to turn them back on, because the callback wrote
    visibility unconditionally."""
    viewer = ViewerModel()
    layer = add_outlines(viewer, labels, min_size=8, cell_diameter=16)
    viewer.camera.zoom = 2.0
    layer.visible = False  # the user turns them off
    viewer.camera.zoom = 3.0  # still legible
    assert not layer.visible
    viewer.camera.zoom = 1.5
    assert not layer.visible


def test_hiding_by_hand_survives_zooming_out_and_back(labels):
    viewer = ViewerModel()
    layer = add_outlines(viewer, labels, min_size=8, cell_diameter=16)
    layer.visible = False
    viewer.camera.zoom = 0.01  # illegible
    viewer.camera.zoom = 2.0  # legible again
    assert not layer.visible


def test_showing_the_layer_by_hand_is_respected(labels):
    viewer = ViewerModel()
    layer = add_outlines(viewer, labels, min_size=8, cell_diameter=16)
    layer.visible = False
    layer.visible = True  # the user turns them back on
    viewer.camera.zoom = 2.0
    assert layer.visible


def test_auto_hide_still_hides_a_layer_the_user_left_on(labels):
    viewer = ViewerModel()
    layer = add_outlines(viewer, labels, min_size=8, cell_diameter=16)
    viewer.camera.zoom = 2.0
    assert layer.visible
    viewer.camera.zoom = 0.01
    assert not layer.visible
