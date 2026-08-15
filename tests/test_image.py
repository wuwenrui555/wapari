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


def test_opens_an_ome_tiff(tmp_path):
    """Both the README and the viewing skill advertise OME-TIFF, and
    tifffile hands back TiffFrame objects for its pages 1..N, which carry
    no description at all."""
    import tifffile

    data = np.arange(2 * 8 * 6, dtype=np.uint16).reshape(2, 8, 6)
    path = tmp_path / "panel.ome.tif"
    tifffile.imwrite(
        path,
        data,
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD8"]}},
    )
    image = open_image(path)
    assert image.channel_names == ["DAPI", "CD8"]
    np.testing.assert_array_equal(np.asarray(image.pyramid("CD8")[0]), data[1])


def test_a_plain_tiff_without_metadata_still_opens(tmp_path):
    """A description that is not XML must not crash the reader."""
    import tifffile

    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((8, 6), dtype=np.uint16), description="hello")
    assert open_image(path).channel_names == ["channel_0"]


def test_an_image_can_be_closed_and_used_as_a_context_manager(qptiff):
    """The qptiff stays open so its levels can be read lazily, so there
    has to be a way to let go of it."""
    with open_image(qptiff[0]) as image:
        assert image.channel_names
    image = open_image(qptiff[0])
    image.close()
    image.close()  # closing twice must not raise


def _bare_zarr(path, channels=None, shape=(3, 8, 8)):
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.create_array("0", shape=shape, chunks=shape, dtype="uint16")
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    if channels is not None:
        root.attrs["omero"] = {"channels": channels}
    return path


def test_ome_zarr_without_channel_labels_still_opens(tmp_path):
    """`label` is optional in NGFF omero; bioformats2raw omits it."""
    path = _bare_zarr(tmp_path / "nolabel.ome.zarr", channels=[{}, {}, {}])
    assert open_image(path).channel_names == ["channel_0", "channel_1", "channel_2"]


def test_ome_zarr_with_too_few_labels_does_not_truncate_the_panel(tmp_path):
    """Reporting a 3-channel image as 1 channel hides two of them."""
    path = _bare_zarr(tmp_path / "short.ome.zarr", channels=[{"label": "DAPI"}])
    assert len(open_image(path).channel_names) == 3


def test_contrast_limits_never_collapse_to_a_single_value(tmp_path):
    """A blank channel gives percentile 0, and napari refuses limits
    whose ends are equal."""
    blank = zarr.open_group(str(tmp_path / "blank.ome.zarr"), mode="w", zarr_format=2)
    blank.create_array("0", shape=(1, 8, 8), chunks=(1, 8, 8), dtype="uint16")
    blank.attrs["omero"] = {"channels": [{"label": "empty"}]}
    blank.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    low, high = open_image(tmp_path / "blank.ome.zarr").contrast_limits("empty")
    assert low < high


def test_a_channel_can_be_named_the_way_the_panel_spells_it_differently(qptiff):
    """The viewing skill tells the agent that a lookup resolves spelling
    differences. It has to be true of the lookup, not only of
    markers.normalize."""
    image = open_image(qptiff[0])
    image.channel_names = ["PD-L1", "CD8"]
    assert image.channel_index("PDL1") == 0
    assert image.channel_index("pd_l1") == 0


def test_a_conjugate_suffix_does_not_hide_a_channel(qptiff):
    image = open_image(qptiff[0])
    image.channel_names = ["FAP-biotin", "CD8"]
    assert image.channel_index("FAP") == 0


def test_an_exact_name_still_wins_over_a_normalised_one(qptiff):
    """Two channels can normalise alike; the literal spelling decides."""
    image = open_image(qptiff[0])
    image.channel_names = ["CD8", "CD-8"]
    assert image.channel_index("CD-8") == 1


def test_a_duplicated_channel_name_is_reported(qptiff):
    """Multi-cycle panels repeat DAPI, and silently taking the first is
    how you end up displaying cycle 1 while believing it is cycle 3."""
    image = open_image(qptiff[0])
    image.channel_names = ["DAPI", "DAPI"]
    with pytest.raises(KeyError, match="appears 2 times"):
        image.channel_index("DAPI")


def test_contrast_limits_do_not_read_a_huge_level(tmp_path):
    """A non-pyramidal slide makes levels[-1] the full plane, and a
    percentile over it would sort two billion pixels."""
    root = zarr.open_group(str(tmp_path / "flat.ome.zarr"), mode="w", zarr_format=2)
    root.create_array("0", shape=(1, 4000, 4000), chunks=(1, 512, 512), dtype="uint16")
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    root.attrs["omero"] = {"channels": [{"label": "DAPI"}]}
    image = open_image(tmp_path / "flat.ome.zarr")
    read = image.contrast_limits("DAPI", max_pixels=10_000)
    assert read[1] >= read[0]


def test_closing_an_image_says_what_breaks(qptiff):
    """Layers added from a closed image fail later, far from the close."""
    image = open_image(qptiff[0])
    image.close()
    with pytest.raises(RuntimeError, match="closed"):
        image.pyramid("DAPI")


def test_a_channel_can_be_taken_by_position(qptiff):
    """When two channels normalise alike the error says to index by
    position, so a position has to be indexable."""
    image = open_image(qptiff[0])
    image.channel_names = ["DAPI", "DAPI"]
    assert image.channel_index(1) == 1
    assert len(image.pyramid(1)) == len(image.levels)
    assert image.contrast_limits(1)[1] >= 0


def test_a_position_outside_the_panel_is_refused(qptiff):
    image = open_image(qptiff[0])
    with pytest.raises(KeyError, match="2 channels"):
        image.channel_index(7)


def test_contrast_limits_also_refuses_a_closed_image(qptiff):
    """pyramid guards; contrast_limits used to succeed on a closed file
    and return a number nobody could trace."""
    image = open_image(qptiff[0])
    image.close()
    with pytest.raises(RuntimeError, match="closed"):
        image.contrast_limits("DAPI")


def test_contrast_limits_reads_a_bounded_sample(tmp_path):
    """The subsample must actually shrink what is sorted, not just
    return something plausible."""
    root = zarr.open_group(str(tmp_path / "flat.ome.zarr"), mode="w", zarr_format=2)
    array = root.create_array(
        "0", shape=(1, 4000, 4000), chunks=(1, 512, 512), dtype="uint16"
    )
    array[:] = 1
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    root.attrs["omero"] = {"channels": [{"label": "DAPI"}]}
    image = open_image(tmp_path / "flat.ome.zarr")
    assert image.sample_size("DAPI", max_pixels=10_000) <= 10_000


def test_a_single_plane_tiff_can_be_displayed(tmp_path):
    """open_image accepts one, so pyramid and contrast_limits have to
    work on it rather than raising about chunk lengths."""
    import tifffile

    path = tmp_path / "plane.tif"
    tifffile.imwrite(path, np.arange(64, dtype=np.uint16).reshape(8, 8))
    image = open_image(path)
    assert len(image.pyramid("channel_0")) == 1
    assert image.contrast_limits("channel_0")[1] > 0


def _ngff05_zarr(path, channels=None, shape=(3, 8, 8), levels=2):
    """A store in the intermediate form: NGFF 0.5 on zarr v3."""
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    datasets = []
    for i in range(levels):
        level = (shape[0], shape[1] // 2**i, shape[2] // 2**i)
        root.create_array(
            str(i), shape=level, dtype="uint16", dimension_names=("c", "y", "x")
        )
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 0.5 * 2**i, 0.5 * 2**i]}
                ],
            }
        )
    ome = {
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
    }
    if channels is not None:
        ome["omero"] = {"channels": channels}
    root.attrs["ome"] = ome
    return path


def test_opens_an_ngff_05_store(tmp_path):
    """0.5 moves everything under an `ome` key, and the whole slide this
    package converted earlier is still 0.4, so both have to open."""
    path = _ngff05_zarr(
        tmp_path / "v05.ome.zarr",
        channels=[{"label": n} for n in ("DAPI", "CD8", "HLA1")],
    )
    image = open_image(path)
    assert image.channel_names == ["DAPI", "CD8", "HLA1"]
    assert len(image.levels) == 2
    assert image.pixel_size_um == 0.5


def test_the_recorded_contrast_window_is_reported(tmp_path):
    """convert.py has been writing these since the beginning and nothing
    has read them; the reader is what makes them mean something."""
    path = _ngff05_zarr(
        tmp_path / "window.ome.zarr",
        channels=[
            {"label": "DAPI", "window": {"start": 0.0, "end": 14737.0}},
            {"label": "CD8"},
            {"label": "HLA1", "window": {"start": 10.0, "end": 300.0}},
        ],
    )
    image = open_image(path)
    assert image.channel_windows == [(0.0, 14737.0), None, (10.0, 300.0)]


def test_the_recorded_channel_colour_is_reported(tmp_path):
    path = _ngff05_zarr(
        tmp_path / "colour.ome.zarr",
        channels=[
            {"label": "DAPI", "color": "0000FF"},
            {"label": "CD8"},
            {"label": "HLA1", "color": "00FF00"},
        ],
    )
    assert open_image(path).channel_colors == ["0000FF", None, "00FF00"]


def test_a_store_with_no_omero_reports_no_defaults(tmp_path):
    path = _ngff05_zarr(tmp_path / "bare.ome.zarr", channels=None)
    image = open_image(path)
    assert image.channel_windows == [None, None, None]
    assert image.channel_colors == [None, None, None]


def test_an_ngff_04_store_still_reports_its_defaults(tmp_path):
    """The whole-slide conversion already on disk is 0.4."""
    path = _bare_zarr(
        tmp_path / "v04.ome.zarr",
        channels=[
            {"label": "DAPI", "color": "FFFFFF", "window": {"start": 0.0, "end": 99.0}},
            {"label": "CD8"},
            {"label": "HLA1"},
        ],
    )
    image = open_image(path)
    assert image.channel_windows[0] == (0.0, 99.0)
    assert image.channel_colors[0] == "FFFFFF"
