"""Tests for wapari.convert.qptiff_to_ome_zarr."""

import numpy as np
import pytest
import tifffile
import zarr

from wapari.convert import qptiff_to_ome_zarr, verify_conversion


@pytest.fixture(scope="module")
def converted(qptiff, tmp_path_factory):
    path, levels, thumbnail = qptiff
    out = tmp_path_factory.mktemp("out") / "synthetic.ome.zarr"
    qptiff_to_ome_zarr(path, out, chunk_size=64, progress=False)
    return out, levels, thumbnail


def test_every_pyramid_level_is_copied_exactly(converted):
    out, levels, _ = converted
    root = zarr.open_group(str(out), mode="r")
    for i, expected in enumerate(levels):
        np.testing.assert_array_equal(np.asarray(root[str(i)]), expected)


def test_dtype_is_preserved(converted):
    out, _, _ = converted
    assert zarr.open_group(str(out), mode="r")["0"].dtype == np.uint16


def test_output_is_chunked_as_requested(converted):
    out, _, _ = converted
    root = zarr.open_group(str(out), mode="r")
    assert root["0"].chunks == (1, 64, 64)


def test_channel_names_written_to_omero_metadata(converted, channel_names):
    out, _, _ = converted
    root = zarr.open_group(str(out), mode="r")
    labels = [ch["label"] for ch in root.attrs["omero"]["channels"]]
    assert labels == channel_names


def test_omero_contrast_window_is_sane(converted):
    out, _, _ = converted
    root = zarr.open_group(str(out), mode="r")
    for ch in root.attrs["omero"]["channels"]:
        window = ch["window"]
        assert 0 <= window["start"] < window["end"] <= 65535


def test_raw_page_xml_preserved_in_attrs(converted, channel_names):
    out, _, _ = converted
    root = zarr.open_group(str(out), mode="r")
    pages = root.attrs["qptiff"]["pages"]
    assert len(pages) == len(channel_names)
    assert "<Biomarker>DAPI</Biomarker>" in pages[0]
    assert "<ExposureTime>11</ExposureTime>" in pages[1]


def test_pixel_size_recorded_in_multiscales_scale(converted, n_levels, pixel_size_um):
    out, _, _ = converted
    root = zarr.open_group(str(out), mode="r")
    datasets = root.attrs["multiscales"][0]["datasets"]
    assert len(datasets) == n_levels
    scales = [d["coordinateTransformations"][0]["scale"] for d in datasets]
    assert scales[0] == [1.0, pixel_size_um, pixel_size_um]
    assert scales[1] == [1.0, pixel_size_um * 2, pixel_size_um * 2]
    assert scales[2] == [1.0, pixel_size_um * 4, pixel_size_um * 4]


def test_auxiliary_series_preserved(converted):
    out, _, thumbnail = converted
    root = zarr.open_group(str(out), mode="r")
    np.testing.assert_array_equal(np.asarray(root["extras"]["thumbnail"]), thumbnail)


def test_single_level_image_converts(tmp_path, channel_names):
    """A series with no sub-resolutions exposes a bare zarr Array rather
    than a group of levels, which is a separate code path."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 65535, (2, 128, 96), dtype=np.uint16)
    path = tmp_path / "flat.qptiff"
    with tifffile.TiffWriter(path) as tw:
        for c, name in enumerate(channel_names):
            tw.write(
                data[c],
                tile=(64, 64),
                metadata=None,
                software="PerkinElmer-QPI",
                description=(
                    "<PerkinElmer-QPI-ImageDescription>"
                    f"<Biomarker>{name}</Biomarker>"
                    "</PerkinElmer-QPI-ImageDescription>"
                ),
            )

    out = qptiff_to_ome_zarr(
        path, tmp_path / "flat.ome.zarr", chunk_size=64, progress=False
    )
    root = zarr.open_group(str(out), mode="r")
    assert len(root.attrs["multiscales"][0]["datasets"]) == 1
    np.testing.assert_array_equal(np.asarray(root["0"]), data)


def test_explicit_channel_names_override_metadata(qptiff, tmp_path):
    path, _, _ = qptiff
    out = tmp_path / "override.ome.zarr"
    qptiff_to_ome_zarr(
        path, out, channel_names=["A", "B"], chunk_size=64, progress=False
    )
    root = zarr.open_group(str(out), mode="r")
    labels = [ch["label"] for ch in root.attrs["omero"]["channels"]]
    assert labels == ["A", "B"]


def test_verify_passes_on_faithful_copy(qptiff, converted):
    path, _, _ = qptiff
    out, _, _ = converted
    verify_conversion(path, out, n_windows=4, window=32)  # must not raise


def test_verify_detects_corrupted_pixels(qptiff, tmp_path):
    path, _, _ = qptiff
    out = tmp_path / "corrupt.ome.zarr"
    qptiff_to_ome_zarr(path, out, chunk_size=64, progress=False)
    root = zarr.open_group(str(out), mode="r+")
    root["0"][:, :64, :64] = 0
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_conversion(path, out, n_windows=64, window=32)


def test_readable_by_ome_zarr_reader(converted, n_levels):
    """The output must be recognized by the ome-zarr ecosystem reader."""
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    out, levels, _ = converted
    nodes = list(Reader(parse_url(str(out)))())
    multiscale = nodes[0]
    assert len(multiscale.data) == n_levels
    assert tuple(multiscale.data[0].shape) == levels[0].shape
