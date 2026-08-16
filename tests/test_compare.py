"""Tests for wapari.compare."""

import numpy as np
import pytest
from napari.components import ViewerModel

from wapari.compare import crosshair, show_comparison


def _ramp(shape=(8, 8), offset=0):
    y, x = np.indices(shape)
    return (y * 100 + x + offset).astype(np.uint16)


def _sources():
    return {
        "CD8": {"raw": _ramp(), "clean": _ramp(offset=1)},
        "CD4": {"raw": _ramp(offset=2), "clean": _ramp(offset=3)},
    }


def test_each_cell_gets_an_image_and_a_crosshair_next_to_it():
    """The grid packs layers two at a time, so the pairing has to be adjacent."""
    viewer = ViewerModel()
    show_comparison(viewer, _sources())

    names = [layer.name for layer in viewer.layers]
    assert names == [
        "CD8 raw",
        "CD8 raw +",
        "CD8 clean",
        "CD8 clean +",
        "CD4 raw",
        "CD4 raw +",
        "CD4 clean",
        "CD4 clean +",
    ]


def test_the_grid_matches_the_shape_of_the_sources():
    viewer = ViewerModel()
    show_comparison(viewer, _sources())

    assert viewer.grid.enabled
    assert viewer.grid.stride == 2  # one image plus its crosshair per cell
    assert viewer.grid.shape == (2, 2)


def test_images_carry_their_row_and_column_for_the_readout():
    viewer = ViewerModel()
    show_comparison(viewer, _sources())

    assert viewer.layers["CD4 clean"].metadata["pixel_values"] == {
        "row": "CD4",
        "column": "clean",
    }
    assert "pixel_values" not in viewer.layers["CD4 clean +"].metadata


def test_a_row_shares_one_pair_of_contrast_limits():
    """Autoscaling each panel separately would make the darker one look as bright
    as the other, which is the one thing a before-and-after view must not do."""
    viewer = ViewerModel()
    show_comparison(viewer, {"CD8": {"raw": _ramp() * 4, "clean": _ramp()}})

    raw = viewer.layers["CD8 raw"].contrast_limits
    clean = viewer.layers["CD8 clean"].contrast_limits
    assert raw == clean


def test_contrast_can_be_left_to_each_panel():
    viewer = ViewerModel()
    show_comparison(
        viewer, {"CD8": {"raw": _ramp() * 4, "clean": _ramp()}}, contrast="each"
    )

    assert viewer.layers["CD8 raw"].contrast_limits != (
        viewer.layers["CD8 clean"].contrast_limits
    )


def test_explicit_contrast_limits_win():
    viewer = ViewerModel()
    show_comparison(viewer, _sources(), contrast={"clean": (0.0, 7.0)})

    assert viewer.layers["CD8 clean"].contrast_limits == [0.0, 7.0]


def test_the_crosshair_is_a_fixed_size_on_screen():
    """In data coordinates the arms must shrink as the zoom grows, or a crosshair
    that looks right zoomed out turns into a slab zoomed in."""
    near, near_width = crosshair(4, 4, zoom=1.0, scale=1.0)
    far, far_width = crosshair(4, 4, zoom=2.0, scale=1.0)

    assert np.allclose(far[:, 1], near[:, 1] / 2)
    assert far_width == pytest.approx(near_width / 2)


def test_the_crosshair_leaves_the_centre_pixel_clear():
    arms, _ = crosshair(4, 6, zoom=1.0, scale=1.0, arm_px=10, gap_px=2)

    starts = arms[:, 0]
    assert (4, 6) not in [tuple(s) for s in starts]
    # the two vertical arms stop short of the centre from either side
    assert starts[0][0] == pytest.approx(4 - 2 - 10)
    assert starts[1][0] == pytest.approx(4 + 2)


def test_the_scale_of_the_layer_is_taken_out_of_the_screen_conversion():
    coarse, _ = crosshair(4, 4, zoom=1.0, scale=1.0, arm_px=10)
    fine, _ = crosshair(4, 4, zoom=1.0, scale=0.5, arm_px=10)

    assert np.allclose(fine[:, 1], coarse[:, 1] * 2)


def test_moving_the_cursor_puts_every_crosshair_on_the_same_pixel():
    viewer = ViewerModel()
    move = show_comparison(viewer, _sources())

    move(viewer, type("E", (), {"position": (3.0, 5.0)})())

    crosses = [layer for layer in viewer.layers if layer.name.endswith(" +")]
    first = crosses[0].data
    assert len(crosses) == 4
    for cross in crosses[1:]:
        assert np.array_equal(cross.data, first)
    assert np.allclose(first[0][0][1], 5.0)  # vertical arms sit on the column


def test_a_position_outside_the_images_leaves_the_crosshair_alone():
    viewer = ViewerModel()
    move = show_comparison(viewer, _sources())
    move(viewer, type("E", (), {"position": (3.0, 5.0)})())
    before = viewer.layers["CD8 raw +"].data.copy()

    move(viewer, type("E", (), {"position": (-5.0, 900.0)})())

    assert np.array_equal(viewer.layers["CD8 raw +"].data, before)


def test_multiscale_sources_are_added_as_pyramids():
    viewer = ViewerModel()
    show_comparison(viewer, {"CD8": {"raw": [_ramp((8, 8)), _ramp((4, 4))]}})

    assert viewer.layers["CD8 raw"].multiscale


def test_a_single_panel_is_a_one_by_one_grid():
    viewer = ViewerModel()
    show_comparison(viewer, {"CD8": {"raw": _ramp()}})

    assert viewer.grid.shape == (1, 1)
    assert [layer.name for layer in viewer.layers] == ["CD8 raw", "CD8 raw +"]


def test_calling_it_again_replaces_the_previous_panel():
    viewer = ViewerModel()
    show_comparison(viewer, _sources())
    show_comparison(viewer, {"CD8": {"raw": _ramp()}})

    assert [layer.name for layer in viewer.layers] == ["CD8 raw", "CD8 raw +"]
    assert len(viewer.mouse_move_callbacks) == 1


class _CountingArray:
    """An array that records whether anything read from it."""

    def __init__(self, data):
        self._data = data
        self.reads = 0
        self.shape = data.shape
        self.dtype = data.dtype
        self.ndim = data.ndim
        self.size = data.size  # napari orders pyramid levels by size

    def __getitem__(self, key):
        self.reads += 1
        return self._data[key]

    def __array__(self, dtype=None, copy=None):
        self.reads += 1
        return np.asarray(self._data, dtype=dtype)


def test_contrast_limits_come_from_the_coarsest_level_of_a_pyramid():
    """Scanning level 0 to pick contrast limits would materialise the whole
    full-resolution plane, which on a slide is hundreds of megabytes per panel
    for a number the smallest level answers just as well."""
    level0 = _CountingArray(_ramp((64, 64)))
    level1 = _ramp((8, 8))
    viewer = ViewerModel()

    show_comparison(
        viewer, {"CD8": {"raw": [level0, level1], "clean": [level0, level1]}}
    )

    assert level0.reads == 0


def test_a_large_single_scale_source_is_sampled_rather_than_read_whole():
    """A one-level source has no coarse level to fall back on, so the sample has
    to come from a handful of windows: materialising the plane to find a minimum
    and a maximum is what made building a panel take twenty seconds."""
    big = _CountingArray(np.zeros((4096, 4096), np.uint16))
    viewer = ViewerModel()

    show_comparison(viewer, {"CD8": {"raw": big}})

    assert 0 < big.reads <= 32


def test_contrast_limits_ignore_a_bright_outlier():
    """A handful of hot pixels must not set the display range.

    Taking the minimum and maximum of the plane lets one outlier stretch the
    limits until the tissue is black, which is what a percentile is for.
    """
    plane = _ramp((64, 64))
    plane[0, 0] = 60000  # a single hot pixel, far above everything else
    viewer = ViewerModel()

    show_comparison(viewer, {"CD8": {"raw": plane}})

    assert viewer.layers["CD8 raw"].contrast_limits[1] < 10000
