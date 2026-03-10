# %%
import warnings
from collections import defaultdict

import napari
import numpy as np
import tifffile
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from wapari.color import assign_bright_colors

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


# %% ========== Functions ==========


def summarize_layers(
    viewer: napari.Viewer, verbose: bool = True
) -> dict[str, list[str]]:
    """
    Get all layer types and names in a napari viewer, grouped by type.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer object
    verbose : bool, optional
        Whether to print detailed information, default is True

    Returns
    -------
    dict[str, list[str]]
        Dictionary where keys are layer type names (e.g., 'Image', 'Labels', 'Shapes'),
        and values are lists of layer names for that type

    Examples
    --------
    >>> layer_dict = summarize_layers(viewer)
    >>> print(layer_dict['Labels'])
    ['mask1', 'mask2']
    """
    layer_dict = defaultdict(list)
    for layer in viewer.layers:
        layer_type = layer.__class__.__name__
        layer_dict[layer_type].append(layer.name)

    # Convert to regular dict for cleaner return type
    layer_dict = dict(layer_dict)

    if verbose:
        for layer_type, layer_names in layer_dict.items():
            print(f"{layer_type}:")
            for name in layer_names:
                print(f'  - "{name}"')
            print()

    return layer_dict


def find_mask_boundaries(
    segmentation_mask: np.ndarray,
    mode: str = "inner",
    connectivity: int = 1,
    background: int = 0,
    thickness: int = 1,
    binary: bool = False,
) -> np.ndarray:
    """
    Convert a segmentation mask to a mask for boundaries.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        An array in which different regions are labeled with different integers.
    mode : str, optional
        The mode of boundary detection. Options are 'inner', 'outer', 'thick',
        and 'subpixel'. Default is 'inner'.
    connectivity : int, optional
        The connectivity defining the neighborhood of a pixel. Default is 1.
    background : int, optional
        The value representing the background in the segmentation mask. Default
        is 0.
    thickness : int, optional
        The thickness of boundaries in pixels. Uses morphological dilation to
        thicken boundaries. Default is 1 (no thickening).
    binary : bool, optional
        If True, return binary boundary mask (True for boundary, False otherwise).
        If False, preserve original label values at boundaries. Default is False.

    Returns
    -------
    np.ndarray
        An array with same shape as `segmentation_mask`.
        If binary=True: bool array with True at boundaries.
        If binary=False: array with same dtype as input, preserving label values
        at boundaries and 0 elsewhere.
    """
    boundaries = find_boundaries(
        segmentation_mask,
        mode=mode,
        connectivity=connectivity,
        background=background,
    )

    # Apply dilation to thicken boundaries if thickness > 1
    if thickness > 1:
        boundaries = dilation(boundaries, disk(thickness - 1))

    if not binary:
        # Ensure correct dtype conversion when preserving label values
        boundaries = boundaries.astype(segmentation_mask.dtype) * segmentation_mask

    return boundaries


def _apply_mapping(mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Apply a mapping to the mask labels using vectorized operations.

    Labels not in the mapping dictionary will be set to 0.

    Parameters
    ----------
    mask : np.ndarray
        Input mask with integer labels
    mapping : dict[int, int]
        Dictionary mapping original labels to new labels

    Returns
    -------
    np.ndarray
        Mapped mask with same shape and dtype as input

    Examples
    --------
    >>> mask = np.array([[1, 2], [3, 4]])
    >>> mapping = {1: 10, 2: 20, 3: 30}
    >>> _apply_mapping(mask, mapping)
    array([[10, 20],
           [30,  0]])
    """
    # Handle empty mask
    if mask.size == 0:
        return mask.copy()

    max_label = mask.max()

    # Handle case where all values are 0 or negative
    if max_label <= 0:
        return np.zeros_like(mask)

    # Create a lookup array initialized with zeros
    lookup = np.zeros(max_label + 1, dtype=mask.dtype)

    # Only update lookup array for keys in mapping
    for original_label, new_label in mapping.items():
        if original_label <= max_label:
            lookup[original_label] = new_label
        else:
            warnings.warn(
                f"Original label {original_label} exceeds max label {max_label} in mask.",
                UserWarning,
                stacklevel=2,
            )

    # Vectorized mapping in a single operation
    mapped_mask = lookup[mask]

    return mapped_mask


class MaskViewer:
    """Helper class for visualizing masks in napari.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance
    mask : np.ndarray
        The segmentation mask to visualize (cell-level labels)
    mapping : dict[int, int], optional
        Mapping from cell labels to cluster labels. If provided,
        the mask will be remapped for visualization.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer instance
    mask_raw : np.ndarray
        The original cell-level mask (reference to input, not copied)
    mask : np.ndarray
        The display mask (cluster-level if mapping provided, otherwise
        same as mask_raw)

    Notes
    -----
    This class does not copy the input mask for performance. Ensure the
    input mask is not modified externally after initialization.
    """

    def __init__(
        self, viewer: napari.Viewer, mask: np.ndarray, mapping: dict[int, int] = None
    ):
        self.viewer = viewer
        self.mask_raw = mask

        if mapping is not None:
            self.mask = _apply_mapping(mask, mapping)
        else:
            self.mask = mask

    def add_mask(self, name: str = "Mask"):
        """Add the mask to the napari viewer.

        Shows cluster-level labels if mapping was provided, otherwise
        shows cell-level labels.
        """
        self.viewer.add_labels(self.mask, name=name)

    def add_boundaries(self, name: str = "Mask Boundaries", thickness: int = 5):
        """Add cell boundaries with cluster labels to the napari viewer.

        Boundaries are computed at cell-level (from mask_raw) but colored
        with cluster-level labels (from mask). This preserves fine boundary
        details while showing cluster assignments.

        Parameters
        ----------
        name : str
            Name of the boundary layer
        thickness : int
            Thickness of boundaries in pixels
        """
        boundaries = find_mask_boundaries(
            self.mask_raw, thickness=thickness, binary=True
        )
        self.viewer.add_labels(self.mask * boundaries, name=name)


# %% ========== Mask ==========

mapping = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60, 7: 70, 10000: 80}


# %%
viewer = napari.Viewer()
# %%
input_f = "/Users/wenruiwu/Downloads/cytassist_image_2.tiff"
output_f = "/Users/wenruiwu/Downloads/Tub972_regions.tiff"

img = tifffile.imread(input_f)

# %%
seg_mask_f = "/Users/wenruiwu/Downloads/segmentation_mask_cell.tiff"
seg_mask = tifffile.imread(seg_mask_f)

viewer.add_labels(seg_mask, name="Labels")

# %%
# directly add the image data to the napari viewer
viewer.add_image(img)

# %%
viewer.layers["Labels"].save(output_f)

# %%
for layer in viewer.layers:
    if isinstance(layer, napari.layers.Labels) and "mask" in layer.name.lower():
        layer.opacity = 1

# %%
layer = "Shapes [1]"

tifffile.imshow(viewer.layers[layer].data[::10, ::10])


# %%
summarize_layers(viewer)

# %% ========== Segmentation ==========
mask = seg_mask

contour = 10

viewer.add_labels(mask, name="Segmentation Mask")


# %%

# %%
# %%
np.random.seed(42)

# Assign a cell type to each cell in the segmentation mask
cell_types = [
    "T",
    "B",
    "Monocyte",
    "Macrophage",
    "Epithelial",
    "Endothelial",
    "Fibroblast",
]

# Get all unique cell labels (excluding background)
unique_labels = np.unique(mask)
unique_labels = unique_labels[unique_labels != 0]  # exclude background

# Randomly assign phenotypes to all cells at once
phenotype_ids = np.random.randint(1, len(cell_types) + 1, size=len(unique_labels))

# Create a lookup array: label -> phenotype_id
# Initialize with zeros, then fill in the assignments
max_label = unique_labels.max()
label_to_phenotype = np.zeros(max_label + 1, dtype=np.int32)
label_to_phenotype[unique_labels] = phenotype_ids

# Vectorized assignment using the lookup array
phenotype_mask = label_to_phenotype[mask]

# %%
viewer.add_labels(phenotype_mask, name="Cell Types")

# %%
import zarr
from ome_zarr.io import parse_url

zarr_path = "/Users/wenruiwu/Downloads/test_8.ome.zarr"
viewer.open(zarr_path + "/labels/segmentation", plugin="napari-ome-zarr")

# 读取 cluster mapping
store = parse_url(zarr_path, mode="r").store
root = zarr.open(store, mode="r")

# %%
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

# 1. 检查版本
# print(f"ome-zarr version: {ome_zarr.__version__}")

# 2. 检查文件结构
path = "/Users/wenruiwu/Downloads/test_8.ome.zarr"
store = parse_url(path, mode="r").store
root = zarr.open(store, mode="r")

print("\n=== Full Tree ===")
print(root.tree())

print("\n=== Root attrs ===")
print(dict(root.attrs))

print("\n=== Labels attrs ===")
if "labels" in root:
    print(dict(root["labels"].attrs))

print("\n=== Segmentation attrs ===")
if "labels" in root and "segmentation" in root["labels"]:
    seg_attrs = dict(root["labels"]["segmentation"].attrs)
    print(seg_attrs)

    # 关键检查
    if "ome" in seg_attrs:
        print("\n❌ PROBLEM: Has 'ome' wrapper (v0.5 format)")
    elif "multiscales" in seg_attrs:
        print("\n✅ Has 'multiscales' at top level (v0.4 format)")
    else:
        print("\n❌ PROBLEM: Missing 'multiscales'")

# 3. 测试 Reader
print("\n=== Reader Test ===")
reader = Reader(parse_url(path))
nodes = list(reader())
print(f"Nodes found: {len(nodes)}")

for i, node in enumerate(nodes):
    print(f"  Node {i}: {node}")
# %%
from wapari.tiff import TiffZarrReader

reader = TiffZarrReader("/Users/wenruiwu/Downloads/marker.ome.tiff")

# %%
markers = [
    "DAPI",
    "CD45",
    "CD3e",
    "CD8",
    "FOXP3",
    "PAX5",
    "CD16",
    "CD163",
    "CD68",
    "CD11b",
    "CD11c",
    "CD15",
    "CD31",
]

for name, img in reader.zimg_dict.items():
    if name not in markers:
        continue

    if name in viewer.layers:
        print(f"Layer '{name}' already exists, skipping...")
        continue

    vmin, vmax = np.percentile(img, [0, 100])

    viewer.add_image(img, name=name, blending="additive", contrast_limits=(vmin, vmax))


# %%
def turn_off_all_layers(viewer: napari.Viewer):
    """Turn off visibility for all layers in the viewer."""
    for layer in tqdm(viewer.layers, desc="Turning off layers", bar_format=TQDM_FORMAT):
        layer.visible = False


turn_off_all_layers(viewer)

# %%
layer = "CD163"
color = "#fe1422"


def set_cmap_color(viewer: napari.Viewer, layer_name: str, color: str):
    """Set colormap color for a given image layer."""
    if layer_name in viewer.layers:
        layer = viewer.layers[layer_name]
        if isinstance(layer, napari.layers.Image):
            layer.colormap = napari.utils.colormaps.Colormap(
                name="custom", colors=["black", color]
            )
        else:
            print(f"Layer '{layer_name}' is not an Image layer.")
    else:
        print(f"Layer '{layer_name}' not found in viewer.")


set_cmap_color(viewer, layer, color)

# %%
colors = assign_bright_colors(markers)
colors_hex = {k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in colors.items()}

for layer_name, color in colors_hex.items():
    set_cmap_color(viewer, layer_name, color)

# %%
