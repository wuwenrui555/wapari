"""Tests for wapari.annotate."""

import numpy as np
import pytest
from napari.components import ViewerModel

from wapari.annotate import (
    DEFAULT_MAX_TEXTURE_SIZE,
    add_labels_for,
    texture_safe_factor,
)


@pytest.fixture
def viewer():
    viewer = ViewerModel()
    viewer.add_image(
        [np.zeros((40, 30), np.uint16), np.zeros((20, 15), np.uint16)],
        name="DAPI",
        multiscale=True,
    )
    return viewer


def test_a_small_image_needs_no_downsampling():
    assert texture_safe_factor((40, 30), limit=16384) == 1


def test_a_shape_at_the_limit_needs_no_downsampling():
    assert texture_safe_factor((16384, 16384), limit=16384) == 1


def test_one_pixel_over_the_limit_needs_downsampling():
    assert texture_safe_factor((16385, 100), limit=16384) == 2


def test_a_whole_slide_is_reduced_until_every_axis_fits():
    """The real case: 48960 over a 16384 limit needs a factor of 3, and
    3 is what keeps the most detail — 4 throws away precision for free."""
    assert texture_safe_factor((48960, 23040), limit=16384) == 3


def test_the_factor_accounts_for_rounding_up():
    """ceil(shape / factor) is what the array will be, so a factor that
    only fits after truncation is not safe."""
    factor = texture_safe_factor((32769, 10), limit=16384)
    assert -(-32769 // factor) <= 16384


def test_labels_are_aligned_to_the_image(viewer):
    labels = add_labels_for(viewer, "DAPI")
    assert tuple(labels.translate) == (0.0, 0.0)
    assert labels.data.shape == (40, 30)  # small image: no downsampling


def test_labels_inherit_the_images_placement(viewer):
    viewer.layers["DAPI"].scale = (2.0, 2.0)
    viewer.layers["DAPI"].translate = (5.0, 7.0)
    labels = add_labels_for(viewer, "DAPI")
    assert tuple(labels.scale) == (2.0, 2.0)
    assert tuple(labels.translate) == (5.0, 7.0)


def test_a_downsampled_layer_still_covers_the_whole_image(viewer):
    labels = add_labels_for(viewer, "DAPI", factor=3)
    covered = np.array(labels.data.shape) * 3
    assert (covered >= (40, 30)).all()
    assert tuple(labels.scale) == (3.0, 3.0)


def test_an_oversized_image_is_not_downsampled(viewer):
    """Going over the GPU texture limit costs memory and nothing else:
    since napari 0.7.0 the polygon preview is correct at any size. Losing
    annotation precision to save memory is the caller's call, not this
    function's, so the layer matches the image."""
    labels = add_labels_for(viewer, "DAPI", shape=(16385, 100))
    assert labels.data.shape == (16385, 100)
    assert tuple(labels.scale) == (1.0, 1.0)


def test_a_texture_safe_layer_can_still_be_asked_for(viewer):
    factor = texture_safe_factor((16385, 100))
    labels = add_labels_for(viewer, "DAPI", shape=(16385, 100), factor=factor)
    assert max(labels.data.shape) <= DEFAULT_MAX_TEXTURE_SIZE
    assert tuple(labels.scale) == (2.0, 2.0)


def test_the_new_layer_is_selected_and_in_polygon_mode(viewer):
    labels = add_labels_for(viewer, "DAPI")
    assert viewer.layers.selection.active is labels
    assert labels.mode == "polygon"


def test_labels_are_small_by_default(viewer):
    assert add_labels_for(viewer, "DAPI").data.dtype == np.uint8


def test_a_wider_dtype_can_be_asked_for(viewer):
    assert add_labels_for(viewer, "DAPI", dtype="uint16").data.dtype == np.uint16


def test_a_second_call_replaces_the_previous_layer(viewer):
    """Otherwise a session accumulates annotations-1, annotations-2 … and
    the user paints into whichever happens to be selected."""
    first = add_labels_for(viewer, "DAPI")
    first.data[0, 0] = 1
    second = add_labels_for(viewer, "DAPI")
    assert [layer.name for layer in viewer.layers].count(second.name) == 1
    assert second.data.sum() == 0


def test_existing_labels_can_be_carried_over(viewer):
    labels = add_labels_for(viewer, "DAPI", data=np.ones((40, 30), np.uint8))
    assert labels.data.sum() == 40 * 30


def test_carried_over_labels_must_match_the_layer_shape(viewer):
    with pytest.raises(ValueError, match="shape"):
        add_labels_for(viewer, "DAPI", data=np.ones((10, 10), np.uint8))


def test_an_unknown_image_names_the_layers_that_exist(viewer):
    with pytest.raises(KeyError, match="DAPI"):
        add_labels_for(viewer, "CD8")


def test_the_default_limit_is_the_common_gpu_maximum():
    assert DEFAULT_MAX_TEXTURE_SIZE == 16384
