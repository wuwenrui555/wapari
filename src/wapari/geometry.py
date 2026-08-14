"""GeoJSON in and out of napari.

napari has no GeoJSON reader, so dragging a QuPath export into a viewer
does nothing. These functions are that reader, plus the writer that makes
napari annotations open in QuPath.

Three conventions have to line up, and all three were measured rather
than assumed:

- **Half a pixel.** napari places a vertex at a pixel's centre; GeoJSON
  and QuPath place it at the pixel's corner, since a pixel occupies
  ``[i, i+1)`` in continuous coordinates. Exported coordinates are the
  napari ones plus 0.5.
- **Axis order.** napari is ``(row, column)``; GeoJSON is ``(x, y)``.
- **Shape type.** A napari ellipse is four vertices plus a separate
  ``shape_type``. GeoJSON has no ellipse, so the type travels in
  ``properties`` or the shape comes back a rectangle.

Rasterising goes through ``Shapes.to_labels``, never through a rasteriser
of our own, so a GeoJSON and the Shapes layer it came from cannot produce
different masks.

There is no ``labels_to_geojson``. Tracing painted pixels gives a
staircase: one region drawn here traced to 32641 vertices where the
polygon that made it had a handful. Rasterising is a one-way step, and
an API that hides that invites a lossy round trip.
"""

import json
import pathlib

import numpy as np

#: What QuPath calls a hand-drawn region. Its cell detections use this
#: too, so it is not a reliable way to tell the two apart.
DEFAULT_OBJECT_TYPE = "annotation"

#: Half a pixel, the offset between napari's centres and GeoJSON's corners.
PIXEL_CORNER_OFFSET = 0.5


def _to_geojson_ring(vertices: np.ndarray) -> list[list[float]]:
    """(row, col) centres to a closed (x, y) corner ring."""
    shifted = np.asarray(vertices, float) + PIXEL_CORNER_OFFSET
    ring = [[float(x), float(y)] for y, x in shifted]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _from_geojson_ring(ring: list) -> np.ndarray:
    """A closed (x, y) corner ring to (row, col) centres."""
    points = list(ring)
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    return np.array([[y, x] for x, y in points], float) - PIXEL_CORNER_OFFSET


def shapes_to_geojson(layer, path: str | pathlib.Path) -> pathlib.Path:
    """Write a Shapes layer as GeoJSON that QuPath can open.

    Per-shape entries in ``layer.features`` become properties. A feature
    named ``classification`` is written the way QuPath expects it, as
    ``{"name": ...}``, so it round-trips as a class rather than a string.
    """
    path = pathlib.Path(path)
    features = []
    table = getattr(layer, "features", None)
    for index, (vertices, shape_type) in enumerate(
        zip(layer.data, layer.shape_type, strict=False)
    ):
        properties: dict = {
            "objectType": DEFAULT_OBJECT_TYPE,
            "shape_type": shape_type,
        }
        if table is not None and len(table.columns):
            for column in table.columns:
                value = table[column].iloc[index]
                value = value.item() if hasattr(value, "item") else value
                if column == "classification":
                    properties["classification"] = {"name": value}
                else:
                    properties[column] = value
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_to_geojson_ring(vertices)],
                },
                "properties": properties,
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=1)
    )
    return path


def _read_features(path: str | pathlib.Path) -> list[dict]:
    document = json.loads(pathlib.Path(path).read_text())
    features = document["features"] if isinstance(document, dict) else list(document)
    if not features:
        raise ValueError(f"{path} holds no features")
    return features


def _class_of(feature: dict) -> str | None:
    classification = feature.get("properties", {}).get("classification")
    if isinstance(classification, dict):
        return classification.get("name")
    return classification


def _parse(features: list[dict], factor: int) -> tuple[list, list, list]:
    vertices, types, properties = [], [], []
    for feature in features:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Polygon":
            raise ValueError(
                f"only Polygon is supported, got {geometry.get('type')!r}; "
                "convert lines and points separately"
            )
        rings = geometry["coordinates"]
        if len(rings) > 1:
            raise ValueError(
                "this polygon has a hole, and a napari Shapes layer cannot "
                "hold one. Rasterise it to Labels instead, where a hole is "
                "just background."
            )
        vertices.append(_from_geojson_ring(rings[0]) / factor)
        properties.append(dict(feature.get("properties", {})))
        types.append(properties[-1].get("shape_type", "polygon"))
    return vertices, types, properties


def _features_table(properties: list[dict]) -> dict:
    table: dict[str, list] = {}
    for entry in properties:
        for key, value in entry.items():
            if key in ("shape_type", "objectType"):
                continue
            if key == "classification":
                value = value.get("name") if isinstance(value, dict) else value
            table.setdefault(key, []).append(value)
    return {
        key: values for key, values in table.items() if len(values) == len(properties)
    }


def geojson_to_shapes(
    path: str | pathlib.Path,
    viewer,
    group_by: str | None = None,
    factor: int = 1,
    name: str = "annotations",
):
    """Read GeoJSON into one Shapes layer, or one per class.

    ``factor`` divides the coordinates, for a viewer whose annotation
    layers are downsampled to stay inside the GPU texture limit.

    Returns the layer, or the list of layers when ``group_by`` is given.
    """
    features = _read_features(path)
    vertices, types, properties = _parse(features, factor)

    if group_by is None:
        layer = viewer.add_shapes(vertices, shape_type=types, name=name)
        table = _features_table(properties)
        if table:
            layer.features = table
        return layer

    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(properties):
        key = _class_of({"properties": entry}) or "unclassified"
        groups.setdefault(key, []).append(index)

    layers = []
    for key, indices in groups.items():
        layer = viewer.add_shapes(
            [vertices[i] for i in indices],
            shape_type=[types[i] for i in indices],
            name=key,
        )
        table = _features_table([properties[i] for i in indices])
        if table:
            layer.features = table
        layers.append(layer)
    return layers


def geojson_to_labels(
    path: str | pathlib.Path,
    viewer,
    shape: tuple[int, int],
    values_from: str | None = None,
    factor: int = 1,
    name: str = "annotations",
    keep_shapes: bool = False,
):
    """Rasterise GeoJSON into a Labels layer.

    By default each polygon gets its own value, numbered by position,
    which is what ``Shapes.to_labels`` does. Pass
    ``values_from="classification"`` to give every polygon of a class the
    same value instead; the mapping is recorded in the layer's metadata.

    ``shape`` is the label array's shape, and ``factor`` the downsampling
    between the GeoJSON's coordinates and that array.
    """
    features = _read_features(path)
    shapes = geojson_to_shapes(path, viewer, factor=factor, name=f"{name}-shapes")
    data = shapes.to_labels(labels_shape=tuple(shape))

    mapping: dict[str, int] = {}
    if values_from is not None:
        if values_from == "classification":
            keys = [_class_of(feature) or "unclassified" for feature in features]
        else:
            keys = [
                feature.get("properties", {}).get(values_from, "unclassified")
                for feature in features
            ]
        mapping = {key: i + 1 for i, key in enumerate(sorted(set(keys)))}
        remapped = np.zeros_like(data)
        for position, key in enumerate(keys, start=1):
            remapped[data == position] = mapping[key]
        data = remapped

    if not keep_shapes:
        viewer.layers.remove(shapes)

    labels = viewer.add_labels(
        data.astype("uint16" if data.max() > 255 else "uint8"),
        name=name,
        scale=(factor, factor),
    )
    labels.metadata["label_values"] = mapping
    labels.metadata["source_geojson"] = str(path)
    return labels
