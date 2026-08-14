"""Tests for wapari.image, lazy access to a multiplexed slide."""

import dask.array as da
import numpy as np
import pytest
import zarr

from wapari.convert import qptiff_to_ome_zarr
from wapari.image import open_image


@pytest.fixture(scope="module")
def zarr_image(qptiff, tmp_path_factory):
    out = tmp_path_factory.mktemp("zarr") / "image.ome.zarr"
    qptiff_to_ome_zarr(qptiff[0], out, chunk_size=64, progress=False)
    return out


def test_opens_a_qptiff_and_reads_its_channel_names(qptiff, channel_names):
    image = open_image(qptiff[0])
    assert image.channel_names == channel_names


def test_opens_an_ome_zarr_and_reads_its_channel_names(zarr_image, channel_names):
    image = open_image(zarr_image)
    assert image.channel_names == channel_names


def test_both_formats_expose_the_same_pyramid(qptiff, zarr_image):
    from_tiff = open_image(qptiff[0])
    from_zarr = open_image(zarr_image)
    assert from_tiff.level_shapes == from_zarr.level_shapes


def test_pixel_size_is_reported(qptiff, pixel_size_um):
    assert open_image(qptiff[0]).pixel_size_um == pytest.approx(pixel_size_um)


def test_opening_reads_no_pixels(qptiff):
    """A 50 GB slide has to open in seconds, so nothing may be loaded
    until a view of it is actually looked at."""
    image = open_image(qptiff[0])
    assert all(isinstance(level, da.Array) for level in image.pyramid("DAPI"))


def test_a_channel_pyramid_is_one_2d_array_per_level(qptiff, qptiff_levels):
    pyramid = open_image(qptiff[0]).pyramid("CD8")
    assert [tuple(level.shape) for level in pyramid] == [
        level.shape[1:] for level in qptiff_levels
    ]


def test_channel_pixels_match_the_source(qptiff, qptiff_levels, channel_names):
    pyramid = open_image(qptiff[0]).pyramid("CD8")
    index = channel_names.index("CD8")
    np.testing.assert_array_equal(np.asarray(pyramid[0]), qptiff_levels[0][index])


def test_an_unknown_channel_names_the_ones_that_exist(qptiff):
    with pytest.raises(KeyError) as raised:
        open_image(qptiff[0]).pyramid("CD99")
    assert "CD8" in str(raised.value)


def test_contrast_limits_come_from_the_smallest_level(qptiff):
    """Percentiles over the full level 0 of a real slide would take
    minutes; the smallest level answers the same question."""
    image = open_image(qptiff[0])
    low, high = image.contrast_limits("DAPI")
    assert low == 0
    smallest = np.asarray(image.pyramid("DAPI")[-1])
    assert high == pytest.approx(np.percentile(smallest, 99.5))


def test_an_image_can_be_closed_and_used_as_a_context_manager(qptiff):
    """The qptiff stays open so its levels can be read lazily, so there
    has to be a way to let go of it."""
    with open_image(qptiff[0]) as image:
        assert image.channel_names
    image = open_image(qptiff[0])
    image.close()
    image.close()  # closing twice must not raise


def test_contrast_limits_never_collapse_to_a_single_value(tmp_path):
    """A blank channel gives percentile 0, and napari refuses limits
    whose ends are equal."""
    blank = zarr.open_group(str(tmp_path / "blank.ome.zarr"), mode="w", zarr_format=2)
    blank.create_array("0", shape=(1, 8, 8), chunks=(1, 8, 8), dtype="uint16")
    blank.attrs["omero"] = {"channels": [{"label": "empty"}]}
    blank.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    low, high = open_image(tmp_path / "blank.ome.zarr").contrast_limits("empty")
    assert low < high
