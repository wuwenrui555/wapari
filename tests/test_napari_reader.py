"""Tests for wapari.napari_reader.

Only the non-interactive path is tested. The dialog is what a person
sees, and the code that decides what to put up is `display.channel_layers`,
tested next door.
"""

import numpy as np
import pytest
import zarr

from wapari import napari_reader
from wapari.napari_reader import napari_get_reader


def _store(path, version="0.5"):
    zarr_format = 3 if version == "0.5" else 2
    root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    datasets = []
    for i, shape in enumerate([(2, 16, 16), (2, 8, 8)]):
        extra = {"dimension_names": ("c", "y", "x")} if zarr_format == 3 else {}
        array = root.create_array(str(i), shape=shape, dtype="uint16", **extra)
        array[:] = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 0.5 * 2**i, 0.5 * 2**i]}
                ],
            }
        )
    multiscales = [
        {
            "name": "probe",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
        }
    ]
    omero = {"channels": [{"label": "DAPI"}, {"label": "CD8"}]}
    if version == "0.5":
        root.attrs["ome"] = {
            "version": "0.5",
            "multiscales": multiscales,
            "omero": omero,
        }
    else:
        root.attrs["multiscales"] = multiscales
        root.attrs["omero"] = omero
    return path


@pytest.fixture
def quiet(monkeypatch):
    """The dialog belongs to a person dragging a file in."""
    monkeypatch.setattr(napari_reader, "INTERACTIVE", False)


def test_an_ome_zarr_is_recognised(tmp_path, quiet):
    assert napari_get_reader(str(_store(tmp_path / "a.ome.zarr"))) is not None


def test_an_ngff_04_store_is_recognised(tmp_path, quiet):
    """The whole-slide conversion already on disk is 0.4."""
    assert napari_get_reader(str(_store(tmp_path / "b.ome.zarr", "0.4"))) is not None


def test_something_that_is_not_a_zarr_is_declined(tmp_path):
    other = tmp_path / "slide.qptiff"
    other.write_bytes(b"")
    assert napari_get_reader(str(other)) is None


def test_a_directory_that_is_not_a_store_is_declined(tmp_path):
    empty = tmp_path / "empty.ome.zarr"
    empty.mkdir()
    assert napari_get_reader(str(empty)) is None


def test_a_list_of_paths_is_declined(tmp_path):
    path = str(_store(tmp_path / "c.ome.zarr"))
    assert napari_get_reader([path, path]) is None


def test_without_the_dialog_one_channel_goes_up(tmp_path, quiet):
    """What README promises: tissue rather than a wall of colour."""
    layers = napari_get_reader(str(_store(tmp_path / "d.ome.zarr")))(
        str(tmp_path / "d.ome.zarr")
    )
    assert len(layers) == 1
    assert layers[0][1]["name"] == "DAPI"


def test_the_layer_keeps_its_levels(tmp_path, quiet):
    path = str(_store(tmp_path / "e.ome.zarr"))
    data, meta, kind = napari_get_reader(path)(path)[0]
    assert kind == "image"
    assert meta["multiscale"]
    assert [tuple(level.shape) for level in data] == [(16, 16), (8, 8)]


def test_cancelling_the_dialog_adds_nothing(tmp_path, monkeypatch):
    """Cancelled is not the same as choosing nothing, but both leave the
    viewer alone."""
    monkeypatch.setattr(napari_reader, "INTERACTIVE", True)
    monkeypatch.setattr(napari_reader, "ask_for_markers", lambda *a, **k: None)
    path = str(_store(tmp_path / "f.ome.zarr"))
    assert napari_get_reader(path)(path) == []


def test_the_dialog_decides_what_goes_up(tmp_path, monkeypatch):
    monkeypatch.setattr(napari_reader, "INTERACTIVE", True)
    monkeypatch.setattr(napari_reader, "ask_for_markers", lambda *a, **k: ["CD8"])
    path = str(_store(tmp_path / "g.ome.zarr"))
    layers = napari_get_reader(path)(path)
    assert [meta["name"] for _, meta, _ in layers] == ["CD8"]


def test_preferring_this_reader_settles_the_tie(tmp_path, monkeypatch):
    """napari's builtin also claims *.zarr and a plugin cannot outrank
    it, so the tie is settled by a preference.

    The pattern has to end in a separator: napari appends one to a
    directory before matching, and every store is a directory.
    """
    from napari.plugins.utils import get_preferred_reader
    from napari.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings.plugins, "extension2reader", {})
    # The settings object is an evented model and guards instance
    # attributes, so `save` is silenced on the class instead.
    monkeypatch.setattr(type(settings), "save", lambda *a, **k: None)

    store = _store(tmp_path / "slide.ome.zarr")
    napari_reader.prefer_wapari()
    assert get_preferred_reader(str(store)) == "wapari"

    other = tmp_path / "slide.qptiff"
    other.write_bytes(b"")
    assert get_preferred_reader(str(other)) is None
