"""
Global constants and configuration values for the inframe project.

This module centralizes all project-wide constants including formatting strings,
configuration parameters, and standardized values used across multiple modules.
It serves as a single source of truth for consistent behavior throughout the framework.

Main Components
---------------
* TQDM_FORMAT: Progress bar format string for tqdm library integration
"""

# Progress bar format string for tqdm library: provides a compact, informative
# display with percentage, visual bar, counts, timing, and rate
#
# Format components:
# - {desc}: Task description text
# - {percentage:3.0f}%: Completion percentage with 3-character width, no decimals
# - {bar:10}: Visual progress bar with 10-character width
# - {n_fmt}/{total_fmt}: Current/total item counts with automatic formatting
# - {elapsed}: Time elapsed since start
# - {remaining}: Estimated time remaining
# - {rate_fmt}: Processing rate (items per second) with automatic unit formatting
TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
