"""Fixtures shared by the tests that need a multiplexed image."""

import numpy as np
import pytest
import tifffile

CHANNEL_NAMES = ["DAPI", "CD8"]
PIXEL_SIZE_UM = 0.5
BASE_SHAPE = (256, 192)  # per-channel (Y, X)
N_LEVELS = 3


@pytest.fixture(scope="session")
def channel_names():
    return list(CHANNEL_NAMES)


@pytest.fixture(scope="session")
def pixel_size_um():
    return PIXEL_SIZE_UM


@pytest.fixture(scope="session")
def n_levels():
    return N_LEVELS


@pytest.fixture(scope="session")
def qptiff_levels():
    """Pixel data for each pyramid level, as (C, Y, X)."""
    rng = np.random.default_rng(0)
    level0 = rng.integers(0, 65535, (2, *BASE_SHAPE), dtype=np.uint16)
    return [level0, level0[:, ::2, ::2], level0[:, ::4, ::4]]


@pytest.fixture(scope="session")
def qptiff_thumbnail():
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (32, 24, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def qptiff(tmp_path_factory, qptiff_levels, qptiff_thumbnail):
    """Write a synthetic file with the page layout of a real qptiff.

    A PerkinElmer qptiff stores its pyramid as flat sequential pages: one
    page per channel at full resolution (each carrying its own
    ``<Biomarker>`` XML), then a thumbnail, then one page per channel for
    every sub-resolution. ``software="PerkinElmer-QPI"`` is what makes
    tifffile group them into a single ``Baseline`` CYX series, and
    ``metadata=None`` keeps tifffile from claiming the file as its own
    "shaped" format, which would take precedence over that grouping.

    Returns the path, the level data and the thumbnail.
    """
    path = tmp_path_factory.mktemp("data") / "synthetic.qptiff"
    px_per_cm = 1e4 / PIXEL_SIZE_UM
    with tifffile.TiffWriter(path) as tw:
        for c, name in enumerate(CHANNEL_NAMES):
            tw.write(
                qptiff_levels[0][c],
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
            qptiff_thumbnail,
            metadata=None,
            description=(
                "<PerkinElmer-QPI-ImageDescription>"
                "<ImageType>Thumbnail</ImageType>"
                "</PerkinElmer-QPI-ImageDescription>"
            ),
        )
        for level in qptiff_levels[1:]:
            for c in range(len(CHANNEL_NAMES)):
                tw.write(
                    level[c],
                    tile=(64, 64),
                    metadata=None,
                    software="PerkinElmer-QPI",
                )
    return path, qptiff_levels, qptiff_thumbnail
