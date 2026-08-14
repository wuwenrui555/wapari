"""Cut regions out of an image using a label mask.

The mask is what a napari Labels layer holds: zero is background and each
positive integer is one region. Two ways to cut are useful and they are
not interchangeable — a bounding box keeps a rectangle of context, while
a polygon crop keeps only what was drawn and replaces the rest.
"""

import pathlib

import numpy as np
import tifffile

MODES = ("polygon", "bbox")


def _axis_layout(image_shape: tuple[int, ...], mask_shape: tuple[int, int]) -> str:
    """Decide where the spatial axes are, by matching them to the mask.

    Nothing is inferred from channel counts: a three-channel image and a
    three-row mask would make that guess wrong in silence.
    """
    if len(image_shape) == 2:
        if image_shape != mask_shape:
            raise ValueError(
                f"image shape {image_shape} does not match mask shape {mask_shape}"
            )
        return "yx"
    if len(image_shape) == 3:
        first = tuple(image_shape[1:]) == mask_shape
        last = tuple(image_shape[:2]) == mask_shape
        if first and last:
            raise ValueError(
                f"image shape {image_shape} is ambiguous against mask shape "
                f"{mask_shape}: the channel axis could be first or last"
            )
        if first:
            return "cyx"
        if last:
            return "yxc"
        raise ValueError(
            f"image shape {image_shape} does not match mask shape {mask_shape} "
            "on either the leading or the trailing axes"
        )
    raise ValueError(f"expected a 2D or 3D image, got shape {image_shape}")


def crop_by_mask(
    image,
    mask: np.ndarray,
    mode: str = "polygon",
    fill: int = 0,
    labels: list[int] | None = None,
    save_dir: str | pathlib.Path | None = None,
) -> dict[int, np.ndarray]:
    """Cut one crop per label out of ``image``.

    Parameters
    ----------
    image : array
        A 2D ``(Y, X)`` image, or 3D with the channel axis either first or
        last. Anything sliceable works, including a lazy zarr or dask
        array; only the cropped region is read.
    mask : numpy.ndarray
        Integer labels over ``(Y, X)``. Zero is background.
    mode : {"polygon", "bbox"}
        ``"polygon"`` keeps only the labelled pixels and replaces the rest
        with ``fill``; ``"bbox"`` keeps the whole bounding box, including
        whatever else falls inside it.
    fill : int
        Value for pixels outside the label in ``"polygon"`` mode. Use 0
        for fluorescence and 255 for 8-bit brightfield, where background
        is white; there is no way to infer which, so it is not guessed.
    labels : list[int], optional
        Which labels to crop. Defaults to every label present.
    save_dir : str or pathlib.Path, optional
        If given, also write ``crop_<label>.tiff`` there, preserving dtype.

    Returns
    -------
    dict[int, numpy.ndarray]
        One crop per label, keyed by label value.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")

    mask = np.asarray(mask)
    layout = _axis_layout(tuple(image.shape), tuple(mask.shape))

    present = [int(value) for value in np.unique(mask) if value != 0]
    if labels is None:
        labels = present
    else:
        missing = [label for label in labels if label not in present]
        if missing:
            raise KeyError(
                f"no such label(s) in the mask: {', '.join(map(str, missing))}; "
                f"it holds {', '.join(map(str, present)) or 'nothing'}"
            )

    dtype = np.dtype(image.dtype)
    if mode == "polygon" and np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if not info.min <= fill <= info.max:
            raise ValueError(
                f"fill value {fill} does not fit the image dtype {dtype}, whose "
                f"range is {info.min} to {info.max}"
            )

    crops: dict[int, np.ndarray] = {}
    for label in labels:
        rows, columns = np.where(mask == label)
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(columns.min()), int(columns.max()) + 1

        # Slice before realizing, so a whole-slide image is never read.
        # np.array, not np.asarray: asarray hands back a view of a numpy
        # input, and filling the outside would then write into the
        # caller's image.
        if layout == "yx":
            region = np.array(image[y0:y1, x0:x1])
        elif layout == "cyx":
            region = np.array(image[:, y0:y1, x0:x1])
        else:
            region = np.array(image[y0:y1, x0:x1, :])

        if mode == "polygon":
            outside = mask[y0:y1, x0:x1] != label
            if layout == "cyx":
                region[:, outside] = fill
            elif layout == "yxc":
                region[outside, :] = fill
            else:
                region[outside] = fill

        crops[label] = region

    if save_dir is not None:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for label, crop in crops.items():
            tifffile.imwrite(save_dir / f"crop_{label}.tiff", crop)

    return crops
