"""Tests for wapari.pixel_values."""

import numpy as np
import pytest
from napari.components import ViewerModel

from wapari.pixel_values import DEFAULT_COLUMN, read_values


def _ramp(shape=(8, 8)):
    """An array whose value equals y * 100 + x, so any read is self-checking."""
    y, x = np.indices(shape)
    return (y * 100 + x).astype(np.uint16)


def test_reads_the_value_under_a_world_position():
    viewer = ViewerModel()
    viewer.add_image(_ramp(), name="probe")

    table = read_values(viewer, (3, 5))

    assert table.rows == ("probe",)
    assert table.columns == (DEFAULT_COLUMN,)
    assert table.values[("probe", DEFAULT_COLUMN)] == 305


def test_honours_scale_and_translate():
    viewer = ViewerModel()
    viewer.add_image(_ramp(), name="probe", scale=(2.0, 2.0), translate=(10.0, 10.0))

    # world (16, 20) is data (3, 5) once the transform is undone
    assert read_values(viewer, (16, 20)).values[("probe", DEFAULT_COLUMN)] == 305


def test_skips_hidden_layers():
    viewer = ViewerModel()
    viewer.add_image(_ramp(), name="shown")
    viewer.add_image(_ramp(), name="hidden", visible=False)

    assert read_values(viewer, (3, 5)).rows == ("shown",)


def test_skips_layers_that_are_not_images():
    viewer = ViewerModel()
    viewer.add_image(_ramp(), name="probe")
    viewer.add_points(np.array([[3.0, 5.0]]), name="dots")

    assert read_values(viewer, (3, 5)).rows == ("probe",)


def test_reads_full_resolution_from_a_multiscale_layer():
    """The displayed level depends on zoom; the value must not.

    napari's own get_value returns the level currently on screen, which makes a
    reading change as the user zooms. Level 0 is the only answer that stays put.
    """
    level0 = _ramp((8, 8))
    level1 = np.zeros((4, 4), np.uint16)  # deliberately not a downsample of level 0
    viewer = ViewerModel()
    viewer.add_image([level0, level1], name="pyramid", multiscale=True)

    assert read_values(viewer, (3, 5)).values[("pyramid", DEFAULT_COLUMN)] == 305


def test_position_outside_the_layer_reads_none():
    viewer = ViewerModel()
    viewer.add_image(_ramp(), name="probe")

    assert read_values(viewer, (99, 99)).values[("probe", DEFAULT_COLUMN)] is None


def test_metadata_pivots_the_table_into_rows_and_columns():
    viewer = ViewerModel()
    for row in ("CD8", "CD4"):
        for column in ("raw", "clean"):
            viewer.add_image(
                _ramp(),
                name=f"{row} {column}",
                metadata={"pixel_values": {"row": row, "column": column}},
            )

    table = read_values(viewer, (3, 5))

    assert table.rows == ("CD8", "CD4")
    assert table.columns == ("raw", "clean")
    assert table.values[("CD4", "clean")] == 305


def test_a_row_missing_a_column_reads_none_rather_than_going_absent():
    viewer = ViewerModel()
    viewer.add_image(
        _ramp(), name="a", metadata={"pixel_values": {"row": "A", "column": "raw"}}
    )
    viewer.add_image(
        _ramp(), name="b", metadata={"pixel_values": {"row": "B", "column": "clean"}}
    )

    table = read_values(viewer, (3, 5))

    assert table.rows == ("A", "B")
    assert table.columns == ("raw", "clean")
    assert table.values[("A", "clean")] is None


def test_layers_with_and_without_metadata_share_one_table():
    viewer = ViewerModel()
    viewer.add_image(
        _ramp(),
        name="paired",
        metadata={"pixel_values": {"row": "CD8", "column": "raw"}},
    )
    viewer.add_image(_ramp(), name="loose")

    table = read_values(viewer, (3, 5))

    assert table.rows == ("CD8", "loose")
    assert table.columns == ("raw", DEFAULT_COLUMN)
    assert table.values[("loose", DEFAULT_COLUMN)] == 305
    assert table.values[("CD8", DEFAULT_COLUMN)] is None


def test_no_visible_layers_gives_an_empty_table():
    viewer = ViewerModel()
    viewer.add_image(_ramp(), name="hidden", visible=False)

    table = read_values(viewer, (3, 5))

    assert table.rows == ()
    assert table.values == {}


@pytest.mark.parametrize("position", [(3.4, 5.4), (2.6, 4.6)])
def test_a_position_inside_a_pixel_reads_that_pixel(position):
    """Both ends of one pixel's extent must read the same value."""
    viewer = ViewerModel()
    viewer.add_image(_ramp(), name="probe")

    assert read_values(viewer, position).values[("probe", DEFAULT_COLUMN)] == 305
