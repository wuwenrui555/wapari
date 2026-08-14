"""Tests for wapari.selection.select_on_show."""

import numpy as np
import pytest
from napari.components import ViewerModel

from wapari.selection import select_on_show


@pytest.fixture
def viewer():
    viewer = ViewerModel()
    for name in ("DAPI", "CD3", "CD68"):
        viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name=name)
    viewer.layers.selection = {viewer.layers["DAPI"]}
    return viewer


def test_showing_a_layer_selects_it(viewer):
    select_on_show(viewer)
    viewer.layers["CD68"].visible = False
    viewer.layers["CD68"].visible = True
    assert viewer.layers.selection.active is viewer.layers["CD68"]


def test_hiding_a_layer_leaves_the_selection_alone(viewer):
    """QuPath's behaviour: hiding is not a statement about what you want
    to work on, so it must not move the selection out from under you."""
    select_on_show(viewer)
    viewer.layers.selection = {viewer.layers["CD3"]}
    viewer.layers["CD68"].visible = False
    assert viewer.layers.selection.active is viewer.layers["CD3"]


def test_hiding_the_selected_layer_keeps_it_selected(viewer):
    select_on_show(viewer)
    viewer.layers["DAPI"].visible = False
    assert viewer.layers.selection.active is viewer.layers["DAPI"]


def test_showing_selects_only_that_layer(viewer):
    """Adjusting contrast acts on the selection, so a leftover second
    member would silently change a layer the user did not touch."""
    select_on_show(viewer)
    viewer.layers.selection = set(viewer.layers)
    viewer.layers["CD3"].visible = False
    viewer.layers["CD3"].visible = True
    assert set(viewer.layers.selection) == {viewer.layers["CD3"]}


def test_layers_added_afterwards_are_covered(viewer):
    select_on_show(viewer)
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="PanCK")
    viewer.layers["PanCK"].visible = False
    viewer.layers["PanCK"].visible = True
    assert viewer.layers.selection.active is viewer.layers["PanCK"]


def test_adding_a_layer_behaves_exactly_as_napari_already_does(viewer):
    """napari selects a layer as it is added. That is its own behaviour,
    not a visibility change, and this must not alter it either way."""
    plain = ViewerModel()
    plain.add_image(np.zeros((4, 4), dtype=np.uint16), name="first")
    plain.add_image(np.zeros((4, 4), dtype=np.uint16), name="second")

    select_on_show(viewer)
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="VIM")

    assert (viewer.layers.selection.active is viewer.layers["VIM"]) == (
        plain.layers.selection.active is plain.layers["second"]
    )


def test_it_can_be_turned_off_again(viewer):
    disconnect = select_on_show(viewer)
    disconnect()
    viewer.layers.selection = {viewer.layers["CD3"]}
    viewer.layers["CD68"].visible = False
    viewer.layers["CD68"].visible = True
    assert viewer.layers.selection.active is viewer.layers["CD3"]


def test_turning_it_off_covers_layers_added_while_it_was_on(viewer):
    disconnect = select_on_show(viewer)
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="Ki67")
    disconnect()
    viewer.layers.selection = {viewer.layers["CD3"]}
    viewer.layers["Ki67"].visible = False
    viewer.layers["Ki67"].visible = True
    assert viewer.layers.selection.active is viewer.layers["CD3"]


def test_enabling_twice_does_not_double_connect(viewer):
    select_on_show(viewer)
    disconnect = select_on_show(viewer)
    disconnect()
    viewer.layers.selection = {viewer.layers["CD3"]}
    viewer.layers["CD68"].visible = False
    viewer.layers["CD68"].visible = True
    assert viewer.layers.selection.active is viewer.layers["CD3"]


def test_a_removed_layer_does_not_keep_firing(viewer):
    select_on_show(viewer)
    layer = viewer.layers["CD68"]
    viewer.layers.remove(layer)
    viewer.layers.selection = {viewer.layers["CD3"]}
    layer.visible = False
    layer.visible = True
    assert viewer.layers.selection.active is viewer.layers["CD3"]


def test_two_viewers_do_not_share_a_connection_record():
    """The record used to be keyed by id(viewer), and a freed viewer's
    address is handed to the next one, which then gets a disconnect for
    a viewer that no longer exists."""
    first = ViewerModel()
    first.add_image(np.zeros((4, 4), np.uint16), name="a")
    select_on_show(first)
    del first

    second = ViewerModel()
    second.add_image(np.zeros((4, 4), np.uint16), name="a")
    second.add_image(np.zeros((4, 4), np.uint16), name="b")
    select_on_show(second)
    second.layers.selection = {second.layers["a"]}
    second.layers["b"].visible = False
    second.layers["b"].visible = True
    assert second.layers.selection.active is second.layers["b"]
