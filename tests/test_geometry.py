"""Tests for wapari.geometry, the GeoJSON side of annotation.

The conventions these tests pin down were measured, not assumed:

- napari puts a vertex at a pixel's centre, GeoJSON and QuPath put it at
  the pixel's corner, so the two differ by exactly half a pixel.
- napari orders coordinates (row, column); GeoJSON orders them (x, y).
- napari's own ``Shapes.to_labels`` numbers shapes by position, ignoring
  any per-shape feature.
"""

import json

import numpy as np
import pytest
from napari.components import ViewerModel

from wapari.geometry import (
    geojson_to_labels,
    geojson_to_shapes,
    shapes_to_geojson,
)

BOX = np.array([[5.0, 3.0], [5.0, 8.0], [7.0, 8.0], [7.0, 3.0]])  # (row, col)


@pytest.fixture
def viewer():
    return ViewerModel()


@pytest.fixture
def shapes(viewer):
    layer = viewer.add_shapes(name="regions")
    layer.add_polygons(BOX)
    return layer


def read(path):
    return json.loads(path.read_text())


def test_export_writes_a_feature_collection(shapes, tmp_path):
    path = shapes_to_geojson(shapes, tmp_path / "r.geojson")
    written = read(path)
    assert written["type"] == "FeatureCollection"
    assert len(written["features"]) == 1
    assert written["features"][0]["geometry"]["type"] == "Polygon"


def test_export_uses_qupath_pixel_corner_coordinates(shapes, tmp_path):
    """QuPath and GeoJSON put a vertex on the pixel corner; napari puts
    it at the centre. Half a pixel, and it is the whole compatibility
    story."""
    ring = read(shapes_to_geojson(shapes, tmp_path / "r.geojson"))["features"][0][
        "geometry"
    ]["coordinates"][0]
    assert [3.5, 5.5] in ring  # (row 5, col 3) -> (x 3.5, y 5.5)


def test_export_orders_coordinates_x_then_y(shapes, tmp_path):
    ring = read(shapes_to_geojson(shapes, tmp_path / "r.geojson"))["features"][0][
        "geometry"
    ]["coordinates"][0]
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    assert max(xs) == 8.5 and max(ys) == 7.5  # columns reach further than rows


def test_export_closes_the_ring(shapes, tmp_path):
    ring = read(shapes_to_geojson(shapes, tmp_path / "r.geojson"))["features"][0][
        "geometry"
    ]["coordinates"][0]
    assert ring[0] == ring[-1]


def test_export_records_the_shape_type(viewer, tmp_path):
    """An ellipse is four vertices plus a type tag. Without the tag it
    comes back a rectangle."""
    layer = viewer.add_shapes(name="e")
    layer.add_ellipses(BOX)
    written = read(shapes_to_geojson(layer, tmp_path / "e.geojson"))
    assert written["features"][0]["properties"]["shape_type"] == "ellipse"


def test_export_uses_qupath_property_names(shapes, tmp_path):
    shapes.features = {"classification": ["Tumour"]}
    props = read(shapes_to_geojson(shapes, tmp_path / "r.geojson"))["features"][0][
        "properties"
    ]
    assert props["objectType"] == "annotation"
    assert props["classification"]["name"] == "Tumour"


def test_export_carries_other_features_through(shapes, tmp_path):
    shapes.features = {"score": [0.75]}
    props = read(shapes_to_geojson(shapes, tmp_path / "r.geojson"))["features"][0][
        "properties"
    ]
    assert props["score"] == 0.75


def test_shapes_round_trip_exactly(viewer, shapes, tmp_path):
    path = shapes_to_geojson(shapes, tmp_path / "r.geojson")
    restored = geojson_to_shapes(path, viewer)
    np.testing.assert_allclose(restored.data[0], shapes.data[0])
    assert restored.shape_type == shapes.shape_type


def test_an_ellipse_survives_the_round_trip(viewer, tmp_path):
    layer = viewer.add_shapes(name="e")
    layer.add_ellipses(BOX)
    restored = geojson_to_shapes(
        shapes_to_geojson(layer, tmp_path / "e.geojson"), viewer
    )
    assert restored.shape_type == ["ellipse"]


def test_import_reads_a_qupath_file(viewer, tmp_path):
    """The shape a real QuPath export takes, from an RCC annotation."""
    path = tmp_path / "qupath.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [3.5, 5.5],
                                    [8.5, 5.5],
                                    [8.5, 7.5],
                                    [3.5, 7.5],
                                    [3.5, 5.5],
                                ]
                            ],
                        },
                        "properties": {
                            "objectType": "annotation",
                            "name": "1",
                            "classification": {"name": "Tumour", "color": [200, 0, 0]},
                        },
                    }
                ],
            }
        )
    )
    layer = geojson_to_shapes(path, viewer)
    np.testing.assert_allclose(layer.data[0], BOX)
    assert list(layer.features["classification"]) == ["Tumour"]


def test_import_can_split_layers_by_class(viewer, tmp_path):
    path = tmp_path / "two.geojson"
    path.write_text(
        json.dumps(_collection([("Tumour", 0), ("Stroma", 20), ("Tumour", 40)]))
    )
    layers = geojson_to_shapes(path, viewer, group_by="classification")
    assert {layer.name for layer in layers} == {"Tumour", "Stroma"}
    assert len(dict((layer.name, layer) for layer in layers)["Tumour"].data) == 2


def test_import_keeps_everything_in_one_layer_by_default(viewer, tmp_path):
    path = tmp_path / "two.geojson"
    path.write_text(json.dumps(_collection([("Tumour", 0), ("Stroma", 20)])))
    layer = geojson_to_shapes(path, viewer)
    assert len(layer.data) == 2


def test_labels_are_numbered_by_position_by_default(viewer, tmp_path):
    path = tmp_path / "three.geojson"
    path.write_text(
        json.dumps(_collection([("Tumour", 0), ("Stroma", 20), ("Tumour", 40)]))
    )
    labels = geojson_to_labels(path, viewer, shape=(60, 60))
    assert sorted(int(v) for v in np.unique(labels.data)) == [0, 1, 2, 3]


def test_labels_can_be_numbered_by_class(viewer, tmp_path):
    """Two tumour polygons share a value; the class becomes the label."""
    path = tmp_path / "three.geojson"
    path.write_text(
        json.dumps(_collection([("Tumour", 0), ("Stroma", 20), ("Tumour", 40)]))
    )
    labels = geojson_to_labels(
        path, viewer, shape=(60, 60), values_from="classification"
    )
    assert sorted(int(v) for v in np.unique(labels.data)) == [0, 1, 2]
    assert labels.metadata["label_values"] == {"Stroma": 1, "Tumour": 2}


def test_the_two_paths_to_labels_agree(viewer, shapes, tmp_path):
    """geojson_to_labels must rasterise through the same napari call that
    a Shapes layer uses, or the two drift."""
    direct = shapes.to_labels(labels_shape=(12, 12))
    path = shapes_to_geojson(shapes, tmp_path / "r.geojson")
    through_geojson = geojson_to_labels(path, viewer, shape=(12, 12))
    np.testing.assert_array_equal(np.asarray(through_geojson.data), direct)


def test_a_mask_survives_labels_to_geojson_to_labels(viewer, tmp_path):
    """The direction that is lossless: raster in, raster out."""
    mask = np.zeros((12, 12), np.uint8)
    mask[5:8, 3:9] = 1
    layer = viewer.add_shapes(name="s")
    layer.add_polygons(BOX)
    path = shapes_to_geojson(layer, tmp_path / "r.geojson")
    back = geojson_to_labels(path, viewer, shape=(12, 12))
    np.testing.assert_array_equal((np.asarray(back.data) > 0).astype(np.uint8), mask)


def test_labels_are_aligned_to_a_downsampled_layer(viewer, tmp_path):
    """Annotation layers are downsampled to stay inside the GPU texture
    limit, so a GeoJSON in full-resolution coordinates has to be divided
    by that factor or it lands three times too far out."""
    path = tmp_path / "r.geojson"
    path.write_text(json.dumps(_collection([("Tumour", 0)], size=30)))
    labels = geojson_to_labels(path, viewer, shape=(20, 20), factor=3)
    rows, cols = np.where(np.asarray(labels.data) > 0)
    assert rows.max() <= 11 and cols.max() <= 11
    assert tuple(labels.scale) == (3.0, 3.0)


def test_an_empty_collection_is_refused(viewer, tmp_path):
    path = tmp_path / "empty.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    with pytest.raises(ValueError, match="no features"):
        geojson_to_shapes(path, viewer)


def test_a_polygon_with_a_hole_says_so(viewer, tmp_path):
    """napari Shapes cannot hold an interior ring, and silently dropping
    one would move a boundary without telling anyone."""
    path = tmp_path / "hole.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                                [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]],
                            ],
                        },
                        "properties": {},
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="hole"):
        geojson_to_shapes(path, viewer)


def _collection(entries, size=10):
    features = []
    for name, offset in entries:
        x0, y0 = float(offset), float(offset)
        x1, y1 = x0 + size, y0 + size
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": name},
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}
