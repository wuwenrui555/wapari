"""Tests for wapari.picker.MarkerSelection.

The Qt dialog is not tested. Everything with behaviour lives here, which
is the point of keeping the dialog a shell over this.
"""

import pytest

from wapari.picker import MarkerSelection

PANEL = ["DAPI", "CD8", "CD3", "Pax5", "HLA1", "p53", "Ki67"]


@pytest.fixture
def selection():
    return MarkerSelection(PANEL)


def test_markers_are_listed_alphabetically_ignoring_case(selection):
    """p53 belongs between Pax5 and Ki67 by name, not after every capital."""
    assert selection.visible() == ["CD3", "CD8", "DAPI", "HLA1", "Ki67", "p53", "Pax5"]


def test_an_empty_pattern_shows_everything(selection):
    selection.pattern = ""
    assert len(selection.visible()) == len(PANEL)


def test_the_pattern_is_a_regular_expression(selection):
    selection.pattern = "^CD"
    assert selection.visible() == ["CD3", "CD8"]


def test_matching_ignores_case(selection):
    """Nobody types the capitalisation of a panel from memory."""
    selection.pattern = "cd"
    assert selection.visible() == ["CD3", "CD8"]


def test_a_pattern_matching_nothing_shows_nothing(selection):
    selection.pattern = "zzz"
    assert selection.visible() == []


def test_an_unfinished_regular_expression_is_not_an_error(selection):
    """A filter is typed one character at a time and spends most of its
    life invalid; raising on the way to a valid one is not an option."""
    selection.pattern = "[CD"
    assert selection.visible() == []
    assert selection.pattern_is_valid is False


def test_a_valid_pattern_reports_itself_valid(selection):
    selection.pattern = "[CD]"
    assert selection.pattern_is_valid is True


def test_filtering_never_changes_what_is_selected(selection):
    selection.toggle("CD8")
    selection.pattern = "zzz"
    assert selection.selected == {"CD8"}


def test_toggle_selects_then_deselects(selection):
    selection.toggle("CD8")
    assert selection.selected == {"CD8"}
    selection.toggle("CD8")
    assert selection.selected == set()


def test_the_header_is_none_when_nothing_is_selected(selection):
    assert selection.header_state() == "none"


def test_the_header_is_all_when_every_visible_marker_is_selected(selection):
    selection.pattern = "^CD"
    selection.toggle("CD3")
    selection.toggle("CD8")
    assert selection.header_state() == "all"


def test_the_header_is_partial_when_some_are(selection):
    selection.toggle("CD8")
    assert selection.header_state() == "partial"


def test_the_header_only_looks_at_visible_markers(selection):
    """CD8 is selected and Ki67 is not, but under this filter only CD8
    is on screen, so the header reads all rather than partial."""
    selection.toggle("CD8")
    selection.pattern = "CD8"
    assert selection.header_state() == "all"


def test_the_header_is_none_when_nothing_matches(selection):
    selection.toggle("CD8")
    selection.pattern = "zzz"
    assert selection.header_state() == "none"


def test_toggle_all_selects_the_visible_markers(selection):
    selection.pattern = "^CD"
    selection.toggle_all()
    assert selection.selected == {"CD3", "CD8"}


def test_toggle_all_clears_them_when_they_are_all_selected(selection):
    selection.pattern = "^CD"
    selection.toggle_all()
    selection.toggle_all()
    assert selection.selected == set()


def test_a_partial_header_goes_to_selected(selection):
    """Both directions discard the partial state, so neither is the safe
    one. Going up matches reading the box as "is everything shown
    selected?" and answering "make it so"."""
    selection.toggle("CD8")
    assert selection.header_state() == "partial"
    selection.toggle_all()
    assert selection.selected == {"CD3", "CD8", "DAPI", "HLA1", "Ki67", "p53", "Pax5"}


def test_toggle_all_leaves_markers_the_filter_hides_alone(selection):
    selection.toggle("Ki67")
    selection.pattern = "^CD"
    selection.toggle_all()
    assert selection.selected == {"Ki67", "CD3", "CD8"}
    selection.toggle_all()
    assert selection.selected == {"Ki67"}


def test_toggle_all_does_nothing_when_nothing_matches(selection):
    selection.toggle("CD8")
    selection.pattern = "zzz"
    selection.toggle_all()
    assert selection.selected == {"CD8"}


def test_the_header_follows_the_filter_without_being_told(selection):
    """Filter to two, select them, then clear the filter: seven are now
    visible and two are selected, so the box draws itself partial. The
    two stay selected. Nothing keeps the box and the filter in step
    because the box is computed rather than stored."""
    selection.pattern = "^CD"
    selection.toggle_all()
    assert selection.header_state() == "all"

    selection.pattern = ""
    assert selection.header_state() == "partial"
    assert selection.selected == {"CD3", "CD8"}


def test_the_chosen_markers_come_back_in_panel_order(selection):
    """Layer order should follow the panel, not the order of clicking."""
    selection.toggle("Ki67")
    selection.toggle("CD3")
    assert selection.chosen() == ["CD3", "Ki67"]
