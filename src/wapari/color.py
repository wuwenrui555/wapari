"""
Color generation and management utilities

This module provides comprehensive color generation and manipulation capabilities
with visually distinct colors, colormap creation, and color interpolation. It emphasizes perceptual uniformity and accessibility in color selection for data visualization
and image analysis workflows.

Main Components
---------------
* generate_distinct_colors: function for creating perceptually distinct color sets
* assign_bright_colors: function for mapping labels to distinct colors
* create_colormap: factory function for custom colormap generation
* DefaultColorMap: class for efficient color interpolation and lookup operations
"""

# %%
import colorsys

import numpy as np


def generate_distinct_colors(
    n: int, saturation: float = 0.7, value: float = 0.95
) -> list[tuple[int, int, int]]:
    """
    Generate n visually distinct colors using HSV color space with golden ratio spacing.

    This function generates colors by using the golden ratio conjugate to space
    hues evenly around the color wheel, ensuring maximum visual distinction between
    adjacent colors. For small numbers of colors, it returns predefined primary
    colors for optimal differentiation.

    Parameters
    ----------
    n : int
        Number of colors to generate. Must be positive.
    saturation : float, default 0.7
        Color saturation level in range [0, 1]. Higher values produce more vivid
        colors. Default value provides good visibility while avoiding overwhelming
        brightness.
    value : float, default 0.95
        Color brightness/value level in range [0, 1]. Higher values produce brighter
        colors. Default near-maximum ensures good visibility on dark backgrounds.

    Returns
    -------
    List[Tuple[int, int, int]]
        List of RGB color tuples with integer values in range [0, 255].
        Colors are ordered by hue progression around the color wheel.

    Notes
    -----
    The algorithm uses the golden ratio (φ - 1 ≈ 0.618) to distribute hues evenly,
    which provides optimal spacing for human color perception. For n ≤ 6, returns
    hand-selected primary and secondary colors that maximize visual distinction.
    """
    colors = []
    golden_ratio = 0.618033988749895  # Golden ratio conjugate for optimal hue spacing

    # Use predefined primary colors for optimal distinction with small datasets
    primary_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    # Return subset of primary colors for small requests
    if n <= len(primary_colors):
        return primary_colors[:n]

    # Use golden ratio method for larger color sets to ensure even distribution
    hue = 0
    for _i in range(n):
        # Convert HSV to RGB using standard color space transformation
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Scale from [0,1] to [0,255] integer range for standard RGB representation
        rgb_int = tuple(int(255 * x) for x in rgb)
        colors.append(rgb_int)
        # Advance hue by golden ratio to maximize distance from previous colors
        hue = (hue + golden_ratio) % 1.0

    return colors


def assign_bright_colors(
    labels: list[str | int],
) -> dict[str | int, tuple[int, int, int]]:
    """
    Assign distinct bright RGB colors to categorical labels for visualization.

    Creates a mapping from each unique label to a visually distinct color using
    the golden ratio spacing algorithm. Ensures good color separation for
    categorical data visualization, similar to QuPath's color assignment but
    with improved perceptual uniformity (https://github.com/qupath/qupath).

    Parameters
    ----------
    labels : List[Union[str, int]]
        List of labels that require distinct color assignments.

    Returns
    -------
    dict[Union[str, int], Tuple[int, int, int]]
        Dictionary mapping each unique label to an RGB color tuple.
        Colors are guaranteed to be visually distinct across all labels.
    """
    n_colors = len(labels)

    # Generate distinct colors using optimized spacing algorithm
    colors = generate_distinct_colors(n_colors)

    return dict(zip(labels, colors, strict=False))


def create_colormap(
    name: str, colors: list[tuple[int, int, int]], n_interpolation: int = 256
) -> "DefaultColorMap":
    """
    Create a continuous colormap from discrete RGB colors with linear interpolation.

    Factory function that constructs a DefaultColorMap object capable of
    interpolating between the provided colors to create smooth color gradients.
    Useful for scientific visualization where smooth color transitions represent
    continuous data values.

    Parameters
    ----------
    name : str
        Descriptive name for the colormap for identification and debugging.
    colors : List[Tuple[int, int, int]]
        List of RGB color tuples (values 0-255) defining the colormap endpoints
        and intermediate control points. Minimum 2 colors required.
    n_interpolation : int, default 256
        Number of discrete colors in the interpolated colormap.
        Higher values provide smoother gradients but use more memory.

    Returns
    -------
    DefaultColorMap
        Colormap object capable of mapping continuous values to interpolated colors.

    Notes
    -----
    Linear interpolation is performed independently on R, G, and B channels.
    The colormap pre-computes all interpolated values for efficient lookup.
    """
    # Extract individual color channels for interpolation
    r = [c[0] for c in colors]
    g = [c[1] for c in colors]
    b = [c[2] for c in colors]
    return DefaultColorMap(name, r, g, b, n_interpolation)


class DefaultColorMap:
    """
    Efficient colormap with pre-computed linear interpolation between RGB control points.

    This class provides continuous color mapping by linearly interpolating between
    a list of RGB colors. It pre-computes all interpolated colors during initialization
    for O(1) color lookup performance, making it suitable for high-frequency color
    mapping operations in scientific visualization.

    Attributes
    ----------
    name : str
        Descriptive name of the colormap.
    r, g, b : np.ndarray
        Arrays containing the red, green, and blue values of control points.
    n_colors : int
        Number of discrete colors in the interpolated colormap.
    """

    def __init__(
        self,
        name: str,
        r: list[int],
        g: list[int],
        b: list[int],
        n_colors: int = 256,
    ):
        """
        Initialize colormap with RGB control points and pre-compute interpolation.

        Parameters
        ----------
        name : str
            Descriptive name for the colormap.
        r : List[int]
            Red channel values (0-255) for colormap control points.
        g : List[int]
            Green channel values (0-255) for colormap control points.
        b : List[int]
            Blue channel values (0-255) for colormap control points.
        n_colors : int, default 256
            Number of discrete colors to pre-compute via interpolation.
            Higher values provide smoother gradients at memory cost.

        Notes
        -----
        All RGB lists must have the same length and contain at least 2 elements.
        Pre-computation occurs during initialization for optimal lookup performance.
        """
        self.name = name
        self.r = np.array(r, dtype=np.int32)
        self.g = np.array(g, dtype=np.int32)
        self.b = np.array(b, dtype=np.int32)
        self.n_colors = n_colors

        # Pre-compute all interpolated colors for O(1) lookup performance
        self._colors = {}
        self._precompute_colors()

    def _precompute_colors(self) -> None:
        """
        Pre-compute linearly interpolated colors for efficient lookup.

        This method performs linear interpolation between control points to generate
        a continuous colormap with n_colors discrete steps. Each RGB channel is
        interpolated independently using the same fractional positions along the
        control point sequence.

        Notes
        -----
        Uses linear interpolation: color = start + (end - start) * fraction
        Results are stored as packed 32-bit integers for memory efficiency.
        """
        # Calculate spacing between control points in interpolated color space
        scale = (len(self.r) - 1) / self.n_colors

        for i in range(self.n_colors):
            # Determine which control points bracket this interpolated color
            ind = int(i * scale)
            residual = (i * scale) - ind

            # Perform linear interpolation for each RGB channel
            r = self.r[ind] + int((self.r[ind + 1] - self.r[ind]) * residual)
            g = self.g[ind] + int((self.g[ind + 1] - self.g[ind]) * residual)
            b = self.b[ind] + int((self.b[ind + 1] - self.b[ind]) * residual)

            # Store as packed integer for memory efficiency
            self._colors[i] = self._pack_rgb(r, g, b)

        # Explicitly set final color to avoid floating-point precision issues
        self._colors[self.n_colors - 1] = self._pack_rgb(
            self.r[-1], self.g[-1], self.b[-1]
        )

    @staticmethod
    def _pack_rgb(r: int, g: int, b: int) -> int:
        """
        Pack separate RGB values into a single 32-bit integer for efficient storage.

        Parameters
        ----------
        r : int
            Red channel value (0-255).
        g : int
            Green channel value (0-255).
        b : int
            Blue channel value (0-255).

        Returns
        -------
        int
            Packed RGB value as 32-bit integer with format 0x00RRGGBB.

        Notes
        -----
        Uses bitwise operations for optimal performance: (R << 16) | (G << 8) | B
        """
        return (r << 16) | (g << 8) | b

    @staticmethod
    def _unpack_rgb(color: int) -> tuple[int, int, int]:
        """
        Unpack a 32-bit packed RGB integer into separate channel values.

        Parameters
        ----------
        color : int
            Packed RGB color value in format 0x00RRGGBB.

        Returns
        -------
        Tuple[int, int, int]
            Tuple containing (red, green, blue) values in range [0, 255].

        Notes
        -----
        Uses bitwise operations and masks to extract individual channels efficiently.
        """
        r = (color >> 16) & 0xFF  # Extract red from bits 16-23
        g = (color >> 8) & 0xFF  # Extract green from bits 8-15
        b = color & 0xFF  # Extract blue from bits 0-7
        return r, g, b

    def get_color(self, value: float, min_value: float, max_value: float) -> int:
        """
        Map a continuous value to an interpolated color from the colormap.

        Parameters
        ----------
        value : float
            The data value to map to a color.
        min_value : float
            Minimum value in the data range, mapped to colormap start.
        max_value : float
            Maximum value in the data range, mapped to colormap end.

        Returns
        -------
        int
            Packed RGB color value corresponding to the input value's position
            within the specified range.

        Notes
        -----
        Values outside the [min_value, max_value] range are clamped to the
        colormap endpoints. If min_value > max_value, the colormap is reversed.
        """
        # Convert continuous value to discrete colormap index
        ind = self._get_ind(value, min_value, max_value)
        return self._colors[ind]

    def _get_ind(self, value: float, min_value: float, max_value: float) -> int:
        """
        Convert a continuous value to a discrete colormap index with range validation.

        Parameters
        ----------
        value : float
            Input value to convert to colormap index.
        min_value : float
            Minimum value of the input range.
        max_value : float
            Maximum value of the input range.

        Returns
        -------
        int
            Colormap index in range [0, n_colors-1] corresponding to value position.

        Notes
        -----
        Handles edge cases: equal min/max values, reversed ranges, and out-of-bounds values.
        Uses rounding for index calculation to minimize quantization artifacts.
        """
        # Normalize min/max order to handle reversed colormaps
        max_val = max(min_value, max_value)
        min_val = min(min_value, max_value)

        # Handle degenerate case where range has zero width
        if max_val == min_val:
            return 0

        # Map value to colormap index with rounding for better quantization
        ind = int(round((value - min_val) / (max_val - min_val) * (self.n_colors - 1)))
        # Clamp index to valid range to handle out-of-bounds values gracefully
        ind = max(0, min(ind, self.n_colors - 1))

        # Reverse index if original range was inverted (min_value > max_value)
        return (self.n_colors - 1 - ind) if min_value > max_value else ind
