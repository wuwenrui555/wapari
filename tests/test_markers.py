"""Tests for wapari.markers, the panel knowledge used to choose channels."""

import pytest

from wapari import markers

REAL_PANEL = [
    "DAPI", "Pax5", "CD8", "CD3", "CD14", "CD4", "FoxP3", "CD15", "CD16",
    "CD68", "CD163", "CD11b", "CD206", "GATA3", "AQP4", "CD56", "CD57",
    "LAG3", "TIM3", "TIGIT", "p53", "BCL6", "SPP1", "HIF1a", "NaATPase",
    "PD1", "PDL1", "IDO1", "Iba1", "CLDN5", "GFAP", "PLVAP", "Glut1",
    "HLA1", "HLADR", "CD31", "CD45RA", "GZMB", "CD47", "ERa", "Podoplanin",
    "PanCK", "VIM", "Ki67", "FAP-biotin",
]  # fmt: skip


def test_dapi_is_nuclear():
    assert "nuclear" in markers.roles("DAPI")


def test_a_lineage_marker_is_recognised():
    assert markers.roles("CD68") == frozenset({"lineage"})


def test_a_marker_can_serve_two_purposes():
    """PanCK draws epithelial boundaries and identifies epithelium; a
    single category per marker would have to lose one of those."""
    assert markers.roles("PanCK") >= {"membrane", "lineage"}


def test_an_unknown_marker_has_no_roles_rather_than_a_guess():
    assert markers.roles("Zzz9") == frozenset()


def test_matching_ignores_case_and_punctuation():
    for name in ["PD-L1", "pdl1", "PD_L1"]:
        assert markers.roles(name) == markers.roles("PDL1")


def test_conjugate_suffixes_do_not_hide_a_marker():
    """Panels label conjugates, e.g. FAP-biotin, but the marker is FAP."""
    assert markers.roles("FAP-biotin") == markers.roles("FAP")
    assert markers.roles("CD8-AF488") == markers.roles("CD8")


def test_segmentation_markers_are_nuclear_plus_boundaries():
    chosen = markers.markers_for(REAL_PANEL, "segmentation")
    assert chosen[0] == "DAPI"  # nuclear first: it is the reference channel
    assert "NaATPase" in chosen
    assert "CD3" not in chosen  # a lineage marker is not a boundary


def test_annotation_markers_are_the_lineage_ones():
    chosen = markers.markers_for(REAL_PANEL, "annotation")
    assert {"CD3", "CD8", "CD68", "Pax5"} <= set(chosen)
    assert "Ki67" not in chosen  # a state, not an identity
    assert "DAPI" not in chosen


def test_other_markers_are_the_functional_ones():
    chosen = markers.markers_for(REAL_PANEL, "other")
    assert {"PD1", "PDL1", "Ki67", "GZMB"} <= set(chosen)
    assert "CD3" not in chosen


def test_marker_order_follows_the_panel_not_the_table():
    """The viewer lists channels in panel order; reordering them would
    make the agent's list hard to compare against the file."""
    panel = ["CD68", "CD3", "DAPI", "CD8"]
    assert markers.markers_for(panel, "annotation") == ["CD68", "CD3", "CD8"]


def test_an_unknown_purpose_is_refused():
    with pytest.raises(ValueError, match="purpose"):
        markers.markers_for(REAL_PANEL, "segmentaton")


def test_unknown_markers_are_reported_for_the_user_to_confirm():
    reported = markers.unknown(["DAPI", "CD3", "NewMarker7", "Zzz9"])
    assert reported == ["NewMarker7", "Zzz9"]


def test_the_real_panel_is_almost_fully_covered():
    """A seeded table that misses most of a panel is not yet useful."""
    missing = markers.unknown(REAL_PANEL)
    assert len(missing) <= 2, f"unclassified: {missing}"


def test_every_table_entry_uses_known_roles():
    for name, roles in markers.MARKER_ROLES.items():
        assert roles, f"{name} has no role"
        assert roles <= markers.ROLES, f"{name} has an unknown role: {roles}"


def test_describe_groups_a_panel_by_purpose():
    described = markers.describe(REAL_PANEL)
    assert described["segmentation"][0] == "DAPI"
    assert "CD3" in described["annotation"]
    assert "PD1" in described["other"]
    assert described["unknown"] == markers.unknown(REAL_PANEL)
