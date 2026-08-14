"""What the markers in a panel are for.

Channel names alone do not say how a channel will be used. Segmentation
needs a nuclear channel and something that draws cell boundaries;
annotation needs the markers that identify a cell type; everything else
reports a state rather than an identity. This module records that for the
markers seen so far, so the same panel is grouped the same way every time.

The table is deliberately incomplete. Anything absent comes back from
:func:`unknown` for an agent to propose and a user to confirm, and a
confirmed answer belongs here afterwards.
"""

import re

#: Every role a marker can carry.
ROLES = frozenset({"nuclear", "membrane", "lineage", "functional"})

#: Which roles serve which purpose. A marker may serve more than one.
PURPOSE_ROLES: dict[str, tuple[str, ...]] = {
    "segmentation": ("nuclear", "membrane"),
    "annotation": ("lineage",),
    "other": ("functional",),
}

_NUCLEAR = frozenset({"nuclear"})
_MEMBRANE = frozenset({"membrane"})
_LINEAGE = frozenset({"lineage"})
_FUNCTIONAL = frozenset({"functional"})
_LINEAGE_MEMBRANE = frozenset({"lineage", "membrane"})

#: Marker roles, keyed by normalized name (see :func:`normalize`).
MARKER_ROLES: dict[str, frozenset[str]] = {
    # Nuclear stains: the reference channel for registration and for
    # every segmentation backend.
    "dapi": _NUCLEAR,
    "hoechst": _NUCLEAR,
    "draq5": _NUCLEAR,
    # Boundary markers. These draw where one cell ends and the next
    # begins, which is what a membrane channel is for in segmentation.
    "naatpase": _MEMBRANE,
    "nakatpase": _MEMBRANE,
    "betacatenin": _MEMBRANE,
    "ecadherin": _LINEAGE_MEMBRANE,
    # Lineage markers that also mark a boundary, so they serve both
    # purposes and appear in both lists.
    "panck": _LINEAGE_MEMBRANE,
    "cd45": _LINEAGE_MEMBRANE,
    "cd31": _LINEAGE_MEMBRANE,
    "cldn5": _LINEAGE_MEMBRANE,
    "glut1": frozenset({"functional", "membrane"}),
    # T cells and their subsets.
    "cd3": _LINEAGE,
    "cd4": _LINEAGE,
    "cd8": _LINEAGE,
    "cd45ra": _LINEAGE,
    "foxp3": _LINEAGE,
    "gata3": _LINEAGE,
    # B cells and germinal centres.
    "pax5": _LINEAGE,
    "cd20": _LINEAGE,
    "cd19": _LINEAGE,
    "bcl6": _LINEAGE,
    # Myeloid, NK and granulocytes.
    "cd14": _LINEAGE,
    "cd15": _LINEAGE,
    "cd16": _LINEAGE,
    "cd11b": _LINEAGE,
    "cd11c": _LINEAGE,
    "cd68": _LINEAGE,
    "cd163": _LINEAGE,
    "cd206": _LINEAGE,
    "cd56": _LINEAGE,
    "cd57": _LINEAGE,
    "iba1": _LINEAGE,
    # Stroma, vessels and brain parenchyma.
    "vim": _LINEAGE,
    "gfap": _LINEAGE,
    "aqp4": _LINEAGE,
    "plvap": _LINEAGE,
    "podoplanin": _LINEAGE,
    "sma": _LINEAGE,
    "fap": _LINEAGE,
    # Checkpoints, activation, proliferation and metabolism: a cell's
    # state rather than what kind of cell it is.
    "pd1": _FUNCTIONAL,
    "pdl1": _FUNCTIONAL,
    "lag3": _FUNCTIONAL,
    "tim3": _FUNCTIONAL,
    "tigit": _FUNCTIONAL,
    "ido1": _FUNCTIONAL,
    "gzmb": _FUNCTIONAL,
    "ki67": _FUNCTIONAL,
    "p53": _FUNCTIONAL,
    "hif1a": _FUNCTIONAL,
    "spp1": _FUNCTIONAL,
    "cd47": _FUNCTIONAL,
    "era": _FUNCTIONAL,
    "hla1": _FUNCTIONAL,
    "hladr": _FUNCTIONAL,
}

# Conjugates and fluorophores are labelling detail, not identity.
_CONJUGATES = (
    "biotin",
    "af[0-9]{3}",
    "alexa[0-9]{3}",
    "fitc",
    "pe",
    "apc",
    "cy[0-9]",
    "opal[0-9]{3}",
)
_CONJUGATE_SUFFIX = re.compile(rf"[-_]({'|'.join(_CONJUGATES)})$", re.IGNORECASE)


def normalize(name: str) -> str:
    """Reduce a channel label to the key used in :data:`MARKER_ROLES`.

    Panels spell the same marker many ways — ``PD-L1``, ``PDL1``,
    ``pd_l1`` — and often append the conjugate, as in ``FAP-biotin``.
    """
    name = _CONJUGATE_SUFFIX.sub("", name.strip())
    return re.sub(r"[^a-z0-9]", "", name.lower())


def roles(name: str) -> frozenset[str]:
    """Return the roles of a marker, or an empty set if it is unknown."""
    return MARKER_ROLES.get(normalize(name), frozenset())


def markers_for(names: list[str], purpose: str) -> list[str]:
    """Return the markers in ``names`` that serve ``purpose``.

    Panel order is preserved so the result can be read against the file's
    own channel list. For segmentation the nuclear channels come first,
    since that is the reference every backend needs.
    """
    if purpose not in PURPOSE_ROLES:
        raise ValueError(
            f"unknown purpose {purpose!r}; expected one of "
            f"{', '.join(sorted(PURPOSE_ROLES))}"
        )
    wanted = set(PURPOSE_ROLES[purpose])
    chosen = [name for name in names if roles(name) & wanted]
    if purpose == "segmentation":
        chosen.sort(key=lambda name: "nuclear" not in roles(name))
    return chosen


def nuclear_channel(names: list[str]) -> str:
    """Return the channel to put up first.

    The table's nuclear entries win. Panels spell nuclear stains in ways
    no table will cover in full — ``Hoechst 33342``, ``DAPI-01``, ``DAPI
    (cycle 2)`` — so an unrecognised panel falls back to its first
    channel. Showing the wrong channel is visible and correctable;
    raising leaves the user with an empty viewer.
    """
    if not names:
        raise ValueError("no channels to choose from")
    for name in names:
        if "nuclear" in roles(name):
            return name
    return names[0]


def boundary_markers(names: list[str]) -> list[str]:
    """Return the markers that draw cell boundaries.

    Separate from :func:`nuclear_channel` because a segmentation backend
    takes one nuclear channel and a set of boundary markers, not a single
    list with the nuclear one buried in it.
    """
    return [name for name in names if "membrane" in roles(name)]


def unknown(names: list[str]) -> list[str]:
    """Return the markers with no entry in the table, in panel order."""
    return [name for name in names if not roles(name)]


def describe(names: list[str]) -> dict[str, list[str]]:
    """Group a panel by purpose, plus whatever is not in the table yet."""
    described = {purpose: markers_for(names, purpose) for purpose in PURPOSE_ROLES}
    described["unknown"] = unknown(names)
    return described
