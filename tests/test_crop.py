"""Tests for wapari.crop.crop_by_mask."""

import numpy as np
import pytest
import tifffile
import zarr

from wapari.crop import crop_by_mask

HEIGHT, WIDTH = 20, 30


@pytest.fixture
def mask():
    """Two labels: a rectangle, and a triangle whose bbox is not full."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    mask[2:6, 3:8] = 1
    for row in range(10, 16):
        mask[row, 20 : 20 + (row - 9)] = 2
    return mask


@pytest.fixture
def image_2d():
    return np.arange(HEIGHT * WIDTH, dtype=np.uint16).reshape(HEIGHT, WIDTH)


def test_bbox_crop_is_the_bounding_box_of_each_label(mask, image_2d):
    crops = crop_by_mask(image_2d, mask, mode="bbox")
    assert set(crops) == {1, 2}
    assert crops[1].shape == (4, 5)
    np.testing.assert_array_equal(crops[1], image_2d[2:6, 3:8])


def test_bbox_crop_keeps_pixels_outside_the_polygon(mask, image_2d):
    """A bounding box is a rectangle; label 2 is a triangle inside it."""
    crop = crop_by_mask(image_2d, mask, mode="bbox")[2]
    assert crop.shape == (6, 6)
    np.testing.assert_array_equal(crop, image_2d[10:16, 20:26])


def test_polygon_crop_blanks_everything_outside_the_label(mask, image_2d):
    crop = crop_by_mask(image_2d, mask, mode="polygon")[2]
    inside = mask[10:16, 20:26] == 2
    assert crop[~inside].tolist() == [0] * int((~inside).sum())
    np.testing.assert_array_equal(crop[inside], image_2d[10:16, 20:26][inside])


def test_polygon_fill_value_is_used_for_the_outside(mask, image_2d):
    crop = crop_by_mask(image_2d, mask, mode="polygon", fill=255)[2]
    outside = mask[10:16, 20:26] != 2
    assert set(np.unique(crop[outside])) == {255}


def test_polygon_is_the_default_mode(mask, image_2d):
    assert np.array_equal(
        crop_by_mask(image_2d, mask)[2],
        crop_by_mask(image_2d, mask, mode="polygon")[2],
    )


def test_the_source_image_is_never_modified(mask, image_2d):
    """np.asarray returns a view of a numpy array, so filling the outside
    would otherwise write straight into the caller's image."""
    before = image_2d.copy()
    crop_by_mask(image_2d, mask, mode="polygon", fill=60000)
    np.testing.assert_array_equal(image_2d, before)


def test_two_crops_of_the_same_region_are_independent(mask, image_2d):
    """Cropping twice must not have the second call see the first's fill."""
    box = crop_by_mask(image_2d, mask, mode="bbox")[2]
    crop_by_mask(image_2d, mask, mode="polygon", fill=60000)
    np.testing.assert_array_equal(box, image_2d[10:16, 20:26])


def test_dtype_is_preserved(mask, image_2d):
    assert crop_by_mask(image_2d, mask)[1].dtype == np.uint16


def test_fill_must_fit_the_image_dtype(mask, image_2d):
    with pytest.raises(ValueError, match="fill"):
        crop_by_mask(image_2d, mask, fill=70000)  # beyond uint16


def test_channel_first_image_is_cropped_on_every_channel(mask):
    image = np.random.default_rng(0).integers(
        0, 1000, (3, HEIGHT, WIDTH), dtype=np.uint16
    )
    crop = crop_by_mask(image, mask, mode="polygon")[2]
    assert crop.shape == (3, 6, 6)
    outside = mask[10:16, 20:26] != 2
    for channel in crop:
        assert set(np.unique(channel[outside])) == {0}


def test_channel_last_rgb_image_is_cropped_on_every_channel(mask):
    image = np.random.default_rng(0).integers(
        0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8
    )
    crop = crop_by_mask(image, mask, mode="polygon", fill=255)[2]
    assert crop.shape == (6, 6, 3)
    outside = mask[10:16, 20:26] != 2
    assert set(np.unique(crop[outside])) == {255}


def test_an_image_whose_axes_do_not_match_the_mask_is_refused(mask):
    with pytest.raises(ValueError, match="shape"):
        crop_by_mask(np.zeros((5, 5), dtype=np.uint16), mask)


def test_an_ambiguous_3d_image_is_refused():
    """A (3, 3, 3) image against a (3, 3) mask could be either layout."""
    square = np.zeros((3, 3), dtype=np.uint16)
    square[1, 1] = 1
    with pytest.raises(ValueError, match="ambiguous"):
        crop_by_mask(np.zeros((3, 3, 3), dtype=np.uint16), square)


def test_only_requested_labels_are_cropped(mask, image_2d):
    assert set(crop_by_mask(image_2d, mask, labels=[2])) == {2}


def test_a_missing_label_is_refused(mask, image_2d):
    with pytest.raises(KeyError, match="7"):
        crop_by_mask(image_2d, mask, labels=[7])


def test_an_empty_mask_produces_nothing(image_2d):
    assert crop_by_mask(image_2d, np.zeros((HEIGHT, WIDTH), np.uint16)) == {}


def test_an_unknown_mode_is_refused(mask, image_2d):
    with pytest.raises(ValueError, match="mode"):
        crop_by_mask(image_2d, mask, mode="polygone")


def test_a_lazy_image_reads_only_the_cropped_region(mask, tmp_path):
    """A whole-slide crop must not pull the slide into memory, so the
    array is sliced before it is realized."""
    store = zarr.open_group(str(tmp_path / "big.zarr"), mode="w", zarr_format=2)
    array = store.create_array(
        "0", shape=(2, HEIGHT, WIDTH), chunks=(1, 4, 4), dtype="uint16"
    )
    array[:] = np.arange(2 * HEIGHT * WIDTH, dtype=np.uint16).reshape(2, HEIGHT, WIDTH)
    crop = crop_by_mask(array, mask, mode="bbox")[1]
    assert isinstance(crop, np.ndarray)
    np.testing.assert_array_equal(crop, np.asarray(array[:, 2:6, 3:8]))


def test_crops_are_written_to_disk_when_a_directory_is_given(mask, image_2d, tmp_path):
    crop_by_mask(image_2d, mask, save_dir=tmp_path)
    written = sorted(p.name for p in tmp_path.glob("*.tiff"))
    assert written == ["crop_1.tiff", "crop_2.tiff"]
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "crop_1.tiff"), image_2d[2:6, 3:8]
    )


def test_written_crops_keep_their_dtype(mask, image_2d, tmp_path):
    crop_by_mask(image_2d, mask, save_dir=tmp_path)
    assert tifffile.imread(tmp_path / "crop_1.tiff").dtype == np.uint16
