"""
TIFF file reading, writing, and pyramidal processing utilities for scientific imaging.

This module provides comprehensive TIFF file handling capabilities optimized for
scientific imaging workflows, particularly for OME-TIFF and QPTIFF formats common
in microscopy and digital pathology. It offers efficient lazy loading, region-of-interest
processing, and pyramidal OME-TIFF generation with multi-threading support.

Main Components
---------------
* TiffZarrReader: Class for efficient reading and slicing of TIFF files with zarr backend
* PyramidWriter: Class for creating pyramidal OME-TIFF files from various input sources
"""

# %%
import concurrent.futures
import itertools
import multiprocessing
import os
import pathlib
import uuid
from typing import Union
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import skimage.transform
import tifffile
import zarr
from tqdm import tqdm

from wapari import constants

TQDM_FORMAT = constants.TQDM_FORMAT

################################################################################
# TIFF Reader
################################################################################


class TiffZarrReader:
    """
    High-performance TIFF file reader with zarr backend for lazy loading and efficient slicing.

    This class provides functionality for reading TIFF files with advanced features
    using zarr as the backend storage format. It supports OME-TIFF and QPTIFF files
    with automatic channel name extraction, enables loading specific regions without
    reading entire images, and provides lazy loading capabilities for memory-efficient
    processing of large files.

    Attributes
    ----------
    zimg : zarr.Array
        Main zarr array containing the entire image data.
    channel_names : list[str]
        List of channel names extracted from metadata or provided by user.
    zimg_dict : dict[str, zarr.Array]
        Dictionary mapping channel names to individual zarr arrays for each channel.

    Notes
    -----
    The class automatically detects file format and extracts appropriate metadata.
    For files without embedded channel names, generic names are generated.
    """

    def __init__(
        self,
        tiff_f: Union[str, pathlib.Path],
        channel_names: list[str] = None,
    ):
        """
        Initialize the TIFF reader with automatic format detection and metadata extraction.

        Parameters
        ----------
        tiff_f : str or pathlib.Path
            Path to the TIFF file to be read.
        channel_names : list[str], optional
            Names of the channels. If None, will be inferred from the file type
            (OME-TIFF or QPTIFF metadata) or generated automatically.

        Raises
        ------
        FileNotFoundError
            If the specified TIFF file does not exist.
        ValueError
            If the TIFF file has unsupported axes configuration or if the number
            of provided channel names doesn't match the number of channels in the file.

        Notes
        -----
        Supported axes configurations are CYX (channels, height, width),
        YXC (height, width, channels), and YX (single channel).
        """
        # Validate tiff_f
        tiff_f = pathlib.Path(tiff_f)
        if not tiff_f.exists():
            raise FileNotFoundError(f"File not found: {tiff_f}")

        # Initialize zarr reader for efficient memory usage
        self.zimg = zarr.open(tifffile.imread(tiff_f, level=0, aszarr=True))

        with tifffile.TiffFile(tiff_f) as tif:
            # Extract channel information based on image axes configuration
            axes = tif.series[0].axes
            supported_axes = ["CYX", "YXC", "YX"]

            if not any(axis in axes for axis in supported_axes):
                raise ValueError(
                    f"Unsupported axes in TIFF file: {axes}. Supported axes are {supported_axes}."
                )

            # Determine number of channels based on axes configuration
            if axes in ["CYX", "YXC"]:
                i_channel = axes.index("C")  # Find channel axis position
                n_channel = self.zimg.shape[i_channel]
            elif axes == "YX":
                n_channel = 1  # Single channel grayscale image

            # Extract or generate channel names based on file format
            if channel_names is None:
                if tif.is_ome:
                    channel_names = self.extract_channel_names_ometiff(tiff_f)
                elif tif.is_qpi:
                    channel_names = self.extract_channel_names_qptiff(tiff_f)
                else:
                    # Generate generic channel names for unsupported formats
                    channel_names = [f"channel_{i}" for i in range(n_channel)]

            # Validate that provided channel names match the number of channels
            if len(channel_names) != n_channel:
                raise ValueError(
                    f"channel_names: Expected {n_channel} channel names, got {len(channel_names)}"
                )
            self.channel_names = channel_names

        # Create dictionary mapping channel names to individual zarr arrays
        # This enables efficient channel-specific access without loading all data
        self.zimg_dict = {
            channel_name: zarr.open(
                tifffile.imread(tiff_f, key=i, level=0, aszarr=True)
            )
            for i, channel_name in enumerate(self.channel_names)
        }

    @classmethod
    def from_ometiff(
        cls,
        tiff_f: Union[str, pathlib.Path],
        markerlist_f: Union[str, pathlib.Path] = None,
    ) -> "TiffZarrReader":
        """
        Initialize a TiffZarrReader specifically for OME-TIFF files.

        Parameters
        ----------
        tiff_f : str or pathlib.Path
            Path to the OME-TIFF file.
        markerlist_f : str or pathlib.Path, optional
            Path to a text file containing channel names (one per line).
            If None, channel names are extracted from OME metadata.

        Returns
        -------
        TiffZarrReader
            Initialized reader instance for the OME-TIFF file.

        Notes
        -----
        When markerlist_f is provided, it overrides the channel names in the
        OME metadata. This is useful when metadata is incorrect or missing.
        """
        # Extract channel names from external file or OME metadata
        if markerlist_f is None:
            channel_names = cls.extract_channel_names_ometiff(tiff_f)
        else:
            with open(markerlist_f) as f:
                channel_names = f.readlines()
                channel_names = [x.strip() for x in channel_names]
        return cls(tiff_f, channel_names)

    @classmethod
    def from_qptiff(
        cls,
        tiff_f: Union[str, pathlib.Path],
        markerlist_f: Union[str, pathlib.Path] = None,
    ) -> "TiffZarrReader":
        """
        Initialize a TiffZarrReader specifically for QPTIFF files.

        Parameters
        ----------
        tiff_f : str or pathlib.Path
            Path to the QPTIFF file.
        markerlist_f : str or pathlib.Path, optional
            Path to a text file containing channel names (one per line).
            If None, channel names are extracted from QPTIFF metadata.

        Returns
        -------
        TiffZarrReader
            Initialized reader instance for the QPTIFF file.

        Notes
        -----
        QPTIFF format stores biomarker information in XML page descriptions.
        When markerlist_f is provided, it overrides the embedded metadata.
        """
        # Extract channel names from external file or QPTIFF metadata
        if markerlist_f is None:
            channel_names = cls.extract_channel_names_qptiff(tiff_f)
        else:
            with open(markerlist_f) as f:
                channel_names = f.readlines()
                channel_names = [x.strip() for x in channel_names]
        return cls(tiff_f, channel_names)

    @staticmethod
    def extract_channel_names_ometiff(path: Union[str, pathlib.Path]) -> list[str]:
        """
        Extract channel names from OME-TIFF metadata.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the OME-TIFF file.

        Returns
        -------
        list[str]
            List of channel names extracted from OME metadata. If names are not
            available in metadata, generates generic names like "channel_0".

        Notes
        -----
        Parses the OME-XML metadata embedded in the TIFF file to extract
        channel information. Falls back to generic naming if "Name" attribute
        is not present in the metadata.
        """
        with tifffile.TiffFile(path) as tif:
            # Parse OME-XML metadata to extract channel information
            ome_metadata = ElementTree.fromstring(tif.ome_metadata)
            ome_channels = ome_metadata.findall(".//{*}Channel")
            metadata = pd.DataFrame([channel.attrib for channel in ome_channels])

            try:
                # Extract channel names from "Name" attribute
                channel_names = metadata.Name.tolist()
            except KeyError:
                # Generate generic names if "Name" attribute is missing
                channel_names = [f"channel_{i}" for i in range(len(metadata))]

        return channel_names

    @staticmethod
    def extract_channel_names_qptiff(path: Union[str, pathlib.Path]) -> list[str]:
        """
        Extract channel names from QPTIFF metadata.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the QPTIFF file.

        Returns
        -------
        list[str]
            List of channel names extracted from QPTIFF page descriptions.
            Uses biomarker names when available, otherwise falls back to
            generic naming.

        Notes
        -----
        QPTIFF stores channel information in XML page descriptions.
        Priority order: Biomarker text > Name text > generic channel names.
        """
        with tifffile.TiffFile(path) as tif:
            channel_names = []
            for i, page in enumerate(tif.series[0].pages):
                # Parse XML page description to extract biomarker information
                biomarker = ElementTree.fromstring(page.description).find("Biomarker")

                if biomarker is not None:
                    channel_name = biomarker.text
                else:
                    # Try alternative "Name" field if biomarker is not available
                    name = ElementTree.fromstring(page.description).find("Name")
                    if name is not None:
                        channel_name = name.text
                    else:
                        # Generate generic name as last resort
                        channel_name = f"channel_{i}"

                channel_names.append(channel_name)

        return channel_names

    def channel_index(self, channels: Union[str, list[str]]) -> Union[int, list[int]]:
        """
        Get the index of a channel or list of channels by name.

        Parameters
        ----------
        channels : str or list[str]
            The channel name or list of channel names to get the index of.

        Returns
        -------
        int or list[int]
            The index of the channel or a list of indices corresponding to
            the input channel names.

        Raises
        ------
        ValueError
            If channel names are not available or if an invalid channel type is provided.

        Notes
        -----
        Useful for converting human-readable channel names to numeric indices
        for array slicing operations.
        """
        if self.channel_names is None:
            raise ValueError("Channel names not found")

        if isinstance(channels, str):
            return self.channel_names.index(channels)
        elif isinstance(channels, list):
            return [self.channel_names.index(ch) for ch in channels]
        else:
            raise ValueError(f"Invalid channel type: {type(channels)}")

    def slice_array(self, ymin: int, ymax: int, xmin: int, xmax: int) -> zarr.Array:
        """
        Extract a rectangular region from the main zarr array without loading entire image.

        Parameters
        ----------
        ymin : int
            Minimum Y coordinate (top boundary) of the region.
        ymax : int
            Maximum Y coordinate (bottom boundary) of the region.
        xmin : int
            Minimum X coordinate (left boundary) of the region.
        xmax : int
            Maximum X coordinate (right boundary) of the region.

        Returns
        -------
        zarr.Array
            Sliced zarr array containing only the specified rectangular region.

        Notes
        -----
        This method provides memory-efficient region extraction by leveraging
        zarr's lazy loading capabilities. Only the requested region is loaded
        into memory, making it suitable for processing large images.
        """
        return self.zimg[ymin:ymax, xmin:xmax]

    def slice_dict(
        self, ymin: int, ymax: int, xmin: int, xmax: int
    ) -> dict[str, zarr.Array]:
        """
        Extract a rectangular region from all channels and return as a dictionary.

        Parameters
        ----------
        ymin : int
            Minimum Y coordinate (top boundary) of the region.
        ymax : int
            Maximum Y coordinate (bottom boundary) of the region.
        xmin : int
            Minimum X coordinate (left boundary) of the region.
        xmax : int
            Maximum X coordinate (right boundary) of the region.

        Returns
        -------
        dict[str, zarr.Array]
            Dictionary mapping channel names to sliced zarr arrays for the
            specified rectangular region.

        Notes
        -----
        Provides channel-specific access to rectangular regions, useful for
        multi-channel analysis workflows where individual channels need to
        be processed separately.
        """
        return {
            channel_name: self.zimg_dict[channel_name][ymin:ymax, xmin:xmax]
            for channel_name in self.channel_names
        }


################################################################################
# Pyramidal OME-TIFF Writer
# https://github.com/labsyspharm/ome-tiff-pyramid-tools/blob/master/pyramid_assemble.py
################################################################################


class PyramidWriter:
    """
    High-performance writer for creating pyramidal OME-TIFF files from various input sources.

    This class creates multi-resolution OME-TIFF files with pyramid structure for
    efficient visualization and analysis of large images. It supports multiple input
    formats, automatic data type conversion, and multi-threaded processing for optimal
    performance.

    Attributes
    ----------
    in_imgs : list[zarr.Array]
        List of input images converted to zarr arrays with uniform data type.
    in_chns : list[str]
        List of channel names corresponding to input images.
    target_shape : tuple[int, int]
        Target shape (height, width) for all images in the pyramid.
    target_dtype : np.dtype
        Unified data type for all images in the pyramid.

    Notes
    -----
    The pyramid structure uses powers-of-2 downsampling with configurable tile sizes.
    Supports both regular images and masks with appropriate downsampling methods.
    """

    def __init__(
        self,
        in_imgs: list[zarr.Array],
        in_chns: list[str],
        target_shape: tuple[int, int],
        target_dtype: np.dtype,
    ):
        """
        Initialize the PyramidWriter with validated input data.

        Parameters
        ----------
        in_imgs : list[zarr.Array]
            List of zarr arrays representing input images.
        in_chns : list[str]
            List of channel names corresponding to input images.
        target_shape : tuple[int, int]
            Target shape (height, width) that all images should conform to.
        target_dtype : np.dtype
            Target data type for consistent pyramid generation.

        Notes
        -----
        All input images are converted to the target data type during initialization
        to ensure consistency across the pyramid levels.
        """
        # Convert all images to uniform data type for consistent processing
        self.in_imgs = [img.astype(target_dtype) for img in in_imgs]
        self.in_chns = in_chns
        self.target_shape = target_shape
        self.target_dtype = target_dtype

    @classmethod
    def from_fs(
        cls,
        input_data: list[Union[str, pathlib.Path]],
        channel_names: list[str] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """
        Create PyramidWriter from a list of file paths.

        Parameters
        ----------
        input_data : list[Union[str, pathlib.Path]]
            A list of file paths to TIFF images that will form the pyramid channels.
        channel_names : list[str], optional
            Names of the channels. If None, generates generic names like "channel_0".
        is_mask : bool, optional
            Whether the images are masks (affects validation and processing). Default False.

        Returns
        -------
        PyramidWriter
            Initialized PyramidWriter instance ready for pyramid generation.

        Raises
        ------
        ValueError
            If the number of channel names doesn't match the number of input files,
            or if image validation fails.

        Notes
        -----
        Supports both 2D and 3D input images. 3D images are split into separate
        channels with indexed naming (e.g., "channel_0_0", "channel_0_1").
        """
        # Generate channel names if not provided
        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(len(input_data))]
        if len(channel_names) != len(input_data):
            raise ValueError(
                f"channel_names: Expected {len(input_data)} channel names, got {len(channel_names)}"
            )

        # Process and validate each input image
        in_imgs = []
        in_chns = []
        for i, path in tqdm(
            enumerate(input_data),
            total=len(input_data),
            desc="Loading images",
            bar_format=TQDM_FORMAT,
        ):
            channel_name = channel_names[i]
            img_in = zarr.open(tifffile.imread(path, level=0, aszarr=True))
            if i == 0:
                target_shape = img_in.shape[-2:]  # Use first image to set target shape

            if img_in.ndim == 2:
                # Handle single 2D image
                shape = img_in.shape
                PyramidWriter._validate_image_2d(
                    shape=shape,
                    dtype=img_in.dtype,
                    target_shape=target_shape,
                    is_mask=is_mask,
                    msg_tag=channel_name,
                )
                in_imgs.append(zarr.array(img_in))
                in_chns.append(channel_name)
            elif img_in.ndim == 3:
                # Handle 3D image by splitting into individual 2D channels
                shape = img_in.shape[1:]
                for j in range(img_in.shape[0]):
                    img = img_in[j]
                    PyramidWriter._validate_image_2d(
                        shape=img.shape,
                        dtype=img.dtype,
                        target_shape=target_shape,
                        is_mask=is_mask,
                        msg_tag=f"{channel_name}_{j}",
                    )
                    in_imgs.append(zarr.array(img))
                    in_chns.append(f"{channel_name}_{j}")
            else:
                raise ValueError(f"{path}: Unsupported dimensions: {img_in.ndim}")

        # Determine target data type as the maximum precision among all images
        target_dtype = max([img.dtype for img in in_imgs])

        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @classmethod
    def from_array(
        cls,
        input_data: Union[np.ndarray, zarr.Array],
        channel_names: list[str] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """
        Create PyramidWriter from a numpy array or zarr array.

        Parameters
        ----------
        input_data : np.ndarray or zarr.Array
            2D (single channel) or 3D (C, H, W) array or zarr array containing image data.
        channel_names : list[str], optional
            Names of the channels. If None, generates generic names.
        is_mask : bool, optional
            Whether the images are masks (affects validation). Default False.

        Returns
        -------
        PyramidWriter
            Initialized PyramidWriter instance ready for pyramid generation.

        Raises
        ------
        ValueError
            If input array dimensions are not 2D or 3D, or if channel names
            don't match the number of channels.

        Notes
        -----
        2D arrays are automatically expanded to 3D with a single channel dimension.
        3D arrays are expected to have channels as the first dimension.
        """
        # Convert 2D array to 3D array with single channel
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, ...]
        elif input_data.ndim == 3:
            pass
        else:
            raise ValueError(
                f"input_data: Expected 2D or 3D array, got shape {input_data.shape}"
            )

        # Generate channel names and validate count
        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(input_data.shape[0])]
        if len(channel_names) != input_data.shape[0]:
            raise ValueError(
                f"channel_names: Expected {input_data.shape[0]} channel names, got {len(channel_names)}"
            )

        # Extract target shape from spatial dimensions
        target_shape = input_data.shape[-2:]

        # Process and validate each channel
        in_imgs = []
        in_chns = []
        for channel_name, img in zip(channel_names, input_data):
            PyramidWriter._validate_image_2d(
                shape=img.shape,
                dtype=img.dtype,
                target_shape=target_shape,
                is_mask=is_mask,
                msg_tag=channel_name,
            )
            in_imgs.append(zarr.array(img))
            in_chns.append(channel_name)

        # Determine target data type as maximum precision among all images
        target_dtype = max([img.dtype for img in in_imgs])

        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @classmethod
    def from_dict(
        cls,
        input_data: dict[str, Union[np.ndarray, zarr.Array]],
        channel_names: list[str] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """
        Create PyramidWriter from a dictionary of named image arrays.

        Parameters
        ----------
        input_data : dict[str, Union[np.ndarray, zarr.Array]]
            Dictionary where keys are channel names and values are numpy arrays
            or zarr arrays containing image data.
        channel_names : list[str], optional
            Names of the channels. If None, uses dictionary keys as channel names.
        is_mask : bool, optional
            Whether the images are masks (affects validation). Default False.

        Returns
        -------
        PyramidWriter
            Initialized PyramidWriter instance ready for pyramid generation.

        Raises
        ------
        ValueError
            If image validation fails, unsupported data types are provided,
            or dimensions are incorrect.

        Notes
        -----
        Supports both 2D and 3D arrays within the dictionary. 3D arrays are
        split into multiple channels with indexed naming.
        """
        # Use dictionary keys as channel names if not provided
        if channel_names is None:
            channel_names = list(input_data.keys())
        if len(channel_names) != len(input_data):
            raise ValueError(
                f"channel_names: Expected {len(input_data)} channel names, got {len(channel_names)}"
            )

        # Extract target shape from first image in dictionary
        target_shape = next(iter(input_data.values())).shape[-2:]

        # Process and validate each image in the dictionary
        in_imgs = []
        in_chns = []
        for channel_name, img_in in zip(channel_names, input_data.values()):
            if isinstance(img_in, (np.ndarray, zarr.Array)):
                if img_in.ndim == 2:
                    # Handle 2D image
                    PyramidWriter._validate_image_2d(
                        shape=img_in.shape,
                        dtype=img_in.dtype,
                        target_shape=target_shape,
                        is_mask=is_mask,
                        msg_tag=channel_name,
                    )
                    in_imgs.append(zarr.array(img_in))
                    in_chns.append(channel_name)
                elif img_in.ndim == 3:
                    # Handle 3D image by splitting into separate channels
                    for i in range(img_in.shape[0]):
                        img = img_in[i]
                        PyramidWriter._validate_image_2d(
                            shape=img.shape,
                            dtype=img.dtype,
                            target_shape=target_shape,
                            is_mask=is_mask,
                            msg_tag=f"{channel_name}_{i}",
                        )
                        in_imgs.append(zarr.array(img))
                        in_chns.append(f"{channel_name}_{i}")
                else:
                    raise ValueError(
                        f"{channel_name}: Unsupported dimensions: {img_in.ndim}"
                    )
            else:
                raise ValueError(f"{channel_name}: Unsupported type: {type(img_in)}")

        # Determine target data type as maximum precision among all images
        target_dtype = max([img.dtype for img in in_imgs])

        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @staticmethod
    def _validate_image_2d(
        shape: tuple[int, int],
        dtype: np.dtype,
        target_shape: tuple[int, int],
        is_mask: bool = False,
        msg_tag: str = None,
    ):
        """
        Validate 2D image properties for pyramid generation.

        Parameters
        ----------
        shape : tuple[int, int]
            Shape of the image to validate (height, width).
        dtype : np.dtype
            Data type of the image.
        target_shape : tuple[int, int]
            Expected shape that the image should match.
        is_mask : bool, optional
            Whether the image is a mask. If True, allows 32-bit integer types. Default False.
        msg_tag : str, optional
            Tag to prepend to error messages for identification. Default None.

        Raises
        ------
        ValueError
            If image validation fails due to unsupported data type or shape mismatch.

        Notes
        -----
        Standard images support uint8 and uint16 data types. Mask images additionally
        support uint32 and int32 for label storage. All images must have identical shapes.
        """
        if msg_tag is not None:
            msg_tag = f"{msg_tag}: "

        # Validate data type based on image type (mask vs regular image)
        if dtype == np.uint32 or dtype == np.int32:
            if not is_mask:
                msg = f"{msg_tag}32-bit images are only supported in is_mask = True"
                raise ValueError(msg)
        elif dtype not in (np.uint8, np.uint16):
            msg = f"{msg_tag}Unsupported dtype: {dtype}"
            raise ValueError(msg)

        # Validate that all images have consistent dimensions
        if shape != target_shape:
            msg = f"{msg_tag}Shape mismatch: expected {target_shape}, got {shape}"
            raise ValueError(msg)

    @staticmethod
    def _create_metadata(pixel_size: float, channel_names: list[str]) -> dict:
        """
        Create OME-TIFF metadata dictionary for embedded image information.

        Parameters
        ----------
        pixel_size : float
            Physical size of pixels in microns. Used for spatial calibration.
        channel_names : list[str]
            Names of the channels in the OME-TIFF file.

        Returns
        -------
        dict
            Dictionary containing OME-TIFF metadata including UUID, physical size,
            and channel names for embedding in the output file.

        Notes
        -----
        The metadata includes a UUID for unique identification and optional
        physical pixel size information for proper spatial calibration.
        """
        # Generate unique identifier for this OME-TIFF file
        metadata = {"UUID": uuid.uuid4().urn}

        # Add pixel size information if provided
        if pixel_size:
            metadata.update(
                {
                    "PhysicalSizeX": pixel_size,
                    "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeY": pixel_size,
                    "PhysicalSizeYUnit": "µm",
                }
            )

        # Add channel names for multi-channel identification
        if channel_names:
            metadata.update(
                {
                    "Channel": {"Name": channel_names},
                }
            )
        return metadata

    @staticmethod
    def _create_tile_generators(
        in_imgs: list[zarr.Array],
        output_f: pathlib.Path,
        num_channels: int,
        tile_size: int,
        is_mask: bool,
        target_dtype: np.dtype,
        target_shape: tuple[int, int],
        num_threads: int,
    ) -> tuple[callable, callable]:
        """
        Create tile generator functions for base and pyramid levels with multi-threading.

        Parameters
        ----------
        in_imgs : list[zarr.Array]
            List of input images as zarr arrays.
        output_f : pathlib.Path
            Path to the output file for reading intermediate results.
        num_channels : int
            Number of channels in the images.
        tile_size : int
            Size of tiles in the pyramid (width and height).
        is_mask : bool
            Whether the images are masks (affects downsampling method).
        target_dtype : np.dtype
            Target data type for all tiles.
        target_shape : tuple[int, int]
            Target shape for the output image.
        num_threads : int
            Number of threads for parallel processing.

        Returns
        -------
        tuple[callable, callable, int, np.ndarray, np.ndarray]
            A tuple containing:
            - Function that generates tiles for the base level
            - Function that generates tiles for pyramid levels
            - Number of pyramid levels
            - Array of shapes for each pyramid level
            - Array of tile counts for each pyramid level

        Notes
        -----
        Uses different downsampling strategies: nearest-neighbor for masks,
        local mean downsampling for regular images. Multi-threading is applied
        to pyramid level generation for performance optimization.
        """
        # Calculate pyramid levels and shapes for efficient multi-resolution access
        num_levels = max(1, int(np.ceil(np.log2(max(target_shape) / tile_size)) + 1))
        factors = 2 ** np.arange(num_levels)  # Powers of 2 for standard pyramid
        shapes = np.ceil(np.array(target_shape) / factors[:, None]).astype(int)
        cshapes = np.ceil(shapes / tile_size).astype(int)  # Number of tiles per level

        # Create thread pool for parallel tile processing
        pool = concurrent.futures.ThreadPoolExecutor(num_threads)

        def tiles0():
            """Generate tiles for the base (highest resolution) level."""
            ts = tile_size
            ch, cw = cshapes[0]  # Number of tiles in height and width
            for c, zimg in enumerate(in_imgs, 1):
                img = zimg[:]  # Load entire image for base level
                for j in range(ch):
                    for i in range(cw):
                        # Extract tile with proper boundary handling
                        tile = img[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                        yield tile
                del img  # Free memory after processing each channel

        def tiles(level):
            """Generate tiles for pyramid levels using downsampling."""
            # Read from previously written pyramid level for progressive downsampling
            with tifffile.TiffFile(output_f, is_ome=False) as tiff_out:
                zimg = zarr.open(tiff_out.series[0].aszarr(level=level - 1))
                ts = tile_size * 2  # Tiles are 2x larger at source level

            def tile(coords):
                """Process individual tile with appropriate downsampling method."""
                c, j, i = coords
                if zimg.ndim == 2:
                    assert c == 0  # Single channel case
                    tile = zimg[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                else:
                    tile = zimg[c, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]

                # Apply different downsampling strategies based on image type
                if is_mask:
                    # Use nearest-neighbor for masks to preserve discrete labels
                    tile = tile[::2, ::2]
                else:
                    # Use local mean for smooth downsampling of regular images
                    tile = skimage.transform.downscale_local_mean(tile, (2, 2))
                    tile = np.round(tile).astype(target_dtype)
                return tile

            # Generate all tile coordinates for this pyramid level
            ch, cw = cshapes[level]
            coords = itertools.product(range(num_channels), range(ch), range(cw))
            # Use thread pool for parallel tile processing
            yield from pool.map(tile, coords)

        return tiles0, tiles, num_levels, shapes, cshapes

    def export_ometiff_pyramid(
        self,
        output_f: Union[str, pathlib.Path],
        pixel_size: float = None,
        tile_size: int = 256,
        is_mask: bool = False,
        num_threads: int = 8,
        overwrite: bool = True,
        verbose: bool = False,
    ):
        """
        Generate and save a pyramidal OME-TIFF file with multi-resolution levels.

        Parameters
        ----------
        output_f : str or pathlib.Path
            Path to the output OME-TIFF file. Will be created with pyramid structure.
        pixel_size : float, optional
            Physical size of pixels in microns. Will be recorded in OME-XML metadata
            for proper spatial calibration. Default None.
        tile_size : int, optional
            Width and height of pyramid tiles in pixels (must be multiple of 16).
            Larger tiles may improve compression but increase memory usage. Default 256.
        is_mask : bool, optional
            Whether images are label masks or binary masks. Enables nearest-neighbor
            downsampling to preserve discrete label values. Default False.
        num_threads : int, optional
            Number of parallel threads for image processing. Set to 0 for automatic
            detection based on available CPUs. Default 8.
        overwrite : bool, optional
            Whether to overwrite existing output file. If False, raises error
            when file exists. Default True.
        verbose : bool, optional
            Whether to display progress information during processing. Default False.

        Raises
        ------
        ValueError
            When input validation fails or tile_size is invalid.
        FileExistsError
            When output file exists and overwrite is False.

        Notes
        -----
        Creates a standard image pyramid with powers-of-2 downsampling. Uses
        Adobe Deflate compression with predictor for optimal file size.
        Multi-threading significantly improves performance for large images.
        """
        # Validate and setup output file path
        output_f = pathlib.Path(output_f)
        if output_f.exists():
            if overwrite:
                if verbose:
                    print(f"Overwriting existing file: {output_f}")
                output_f.unlink()
            else:
                raise FileExistsError(f"Output file already exists: {output_f}")

        # Configure thread pool size for optimal performance
        if num_threads == 0:
            # Auto-detect number of available CPU cores
            if hasattr(os, "sched_getaffinity"):
                num_threads = len(os.sched_getaffinity(0))  # Linux/Unix systems
            else:
                num_threads = multiprocessing.cpu_count()  # Fallback for other systems
            print(f"Using {num_threads} worker threads")

        # Configure tifffile library thread limits for optimal resource usage
        tifffile.TIFF.MAXWORKERS = num_threads
        tifffile.TIFF.MAXIOWORKERS = num_threads * 5  # Higher I/O worker count

        # Create OME-TIFF metadata for proper file identification
        metadata = PyramidWriter._create_metadata(
            pixel_size=pixel_size,
            channel_names=self.in_chns,
        )
        num_channels = len(self.in_chns)

        # Create tile generators for efficient pyramid construction
        (
            tiles0,
            tiles,
            num_levels,
            shapes,
            cshapes,
        ) = PyramidWriter._create_tile_generators(
            in_imgs=self.in_imgs,
            output_f=output_f,
            num_channels=num_channels,
            tile_size=tile_size,
            is_mask=is_mask,
            target_dtype=self.target_dtype,
            target_shape=self.target_shape,
            num_threads=num_threads,
        )

        # Initialize progress tracking for user feedback
        pbar = tqdm(
            total=sum(tile_shape[0] * tile_shape[1] for tile_shape in cshapes),
            desc="Writing tiles",
            bar_format=TQDM_FORMAT,
            disable=not verbose,
        )

        # Write multi-resolution pyramid with optimal compression settings
        with tifffile.TiffWriter(output_f, ome=True, bigtiff=True) as writer:
            for level, shape in enumerate(shapes):
                if level == 0:
                    # Write base level with full metadata and SubIFDs for pyramid
                    writer.write(
                        data=tiles0(),
                        shape=(num_channels,) + tuple(shape),
                        subifds=num_levels - 1,  # Reserve space for pyramid levels
                        dtype=self.target_dtype,
                        tile=(tile_size, tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                        metadata=metadata,
                    )
                    pbar.update(cshapes[level][0] * cshapes[level][1])
                else:
                    # Write pyramid levels as SubIFDs
                    writer.write(
                        data=tiles(level),
                        shape=(num_channels,) + tuple(shape),
                        subfiletype=1,  # Mark as pyramid level
                        dtype=self.target_dtype,
                        tile=(tile_size, tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                    )
                    pbar.update(cshapes[level][0] * cshapes[level][1])
        pbar.close()
