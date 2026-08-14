"""Tests for wapari.convert.qptiff_to_ome_zarr."""

import numpy as np
import pytest
import tifffile
import zarr

from wapari.convert import qptiff_to_ome_zarr, verify_conversion

CHANNEL_NAMES = ["DAPI", "CD8"]
PIXEL_SIZE_UM = 0.5
BASE_SHAPE = (256, 192)  # per-channel (Y, X)
N_LEVELS = 3


@pytest.fixture(scope="module")
def qptiff(tmp_path_factory):
    """Write a synthetic file with the page layout of a real qptiff.

    A PerkinElmer qptiff stores its pyramid as flat sequential pages:
    one page per channel at full resolution (each carrying its own
    ``<Biomarker>`` XML), then a thumbnail, then one page per channel
    for every sub-resolution. ``software="PerkinElmer-QPI"`` is what
    makes tifffile group them into a single ``Baseline`` CYX series, and
    ``metadata=None`` keeps tifffile from claiming the file as its own
    "shaped" format, which would take precedence over that grouping.
    """
    rng = np.random.default_rng(0)
    levels = [rng.integers(0, 65535, (2, *BASE_SHAPE), dtype=np.uint16)]
    levels.append(levels[0][:, ::2, ::2])
    levels.append(levels[0][:, ::4, ::4])
    thumbnail = rng.integers(0, 255, (32, 24, 3), dtype=np.uint8)

    path = tmp_path_factory.mktemp("data") / "synthetic.qptiff"
    px_per_cm = 1e4 / PIXEL_SIZE_UM
    with tifffile.TiffWriter(path) as tw:
        for c, name in enumerate(CHANNEL_NAMES):
            tw.write(
                levels[0][c],
                tile=(64, 64),
                metadata=None,
                software="PerkinElmer-QPI",
                description=(
                    "<PerkinElmer-QPI-ImageDescription>"
                    "<ImageType>FullResolution</ImageType>"
                    f"<Biomarker>{name}</Biomarker>"
                    f"<ExposureTime>{10 + c}</ExposureTime>"
                    "</PerkinElmer-QPI-ImageDescription>"
                ),
                resolution=(px_per_cm, px_per_cm),
                resolutionunit="CENTIMETER",
            )
        tw.write(
            thumbnail,
            metadata=None,
            description=(
                "<PerkinElmer-QPI-ImageDescription>"
                "<ImageType>Thumbnail</ImageType>"
                "</PerkinElmer-QPI-ImageDescription>"
            ),
        )
        for level in levels[1:]:
            for c in range(len(CHANNEL_NAMES)):
                tw.write(
                    level[c],
                    tile=(64, 64),
                    metadata=None,
                    software="PerkinElmer-QPI",
                )
    return path, levels, thumbnail


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


def test_channel_names_written_to_omero_metadata(converted):
    out, _, _ = converted
    root = zarr.open_group(str(out), mode="r")
    labels = [ch["label"] for ch in root.attrs["omero"]["channels"]]
    assert labels == CHANNEL_NAMES


def test_omero_contrast_window_is_sane(converted):
    out, _, _ = converted
    root = zarr.open_group(str(out), mode="r")
    for ch in root.attrs["omero"]["channels"]:
        window = ch["window"]
        assert 0 <= window["start"] < window["end"] <= 65535


def test_raw_page_xml_preserved_in_attrs(converted):
    out, _, _ = converted
    root = zarr.open_group(str(out), mode="r")
    pages = root.attrs["qptiff"]["pages"]
    assert len(pages) == len(CHANNEL_NAMES)
    assert "<Biomarker>DAPI</Biomarker>" in pages[0]
    assert "<ExposureTime>11</ExposureTime>" in pages[1]


def test_pixel_size_recorded_in_multiscales_scale(converted):
    out, _, _ = converted
    root = zarr.open_group(str(out), mode="r")
    datasets = root.attrs["multiscales"][0]["datasets"]
    assert len(datasets) == N_LEVELS
    scales = [d["coordinateTransformations"][0]["scale"] for d in datasets]
    assert scales[0] == [1.0, PIXEL_SIZE_UM, PIXEL_SIZE_UM]
    assert scales[1] == [1.0, PIXEL_SIZE_UM * 2, PIXEL_SIZE_UM * 2]
    assert scales[2] == [1.0, PIXEL_SIZE_UM * 4, PIXEL_SIZE_UM * 4]


def test_auxiliary_series_preserved(converted):
    out, _, thumbnail = converted
    root = zarr.open_group(str(out), mode="r")
    np.testing.assert_array_equal(np.asarray(root["extras"]["thumbnail"]), thumbnail)


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


def test_readable_by_ome_zarr_reader(converted):
    """The output must be recognized by the ome-zarr ecosystem reader."""
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    out, levels, _ = converted
    nodes = list(Reader(parse_url(str(out)))())
    multiscale = nodes[0]
    assert len(multiscale.data) == N_LEVELS
    assert tuple(multiscale.data[0].shape) == levels[0].shape
