"""Choosing which markers to put on screen.

Two pieces. :class:`MarkerSelection` is the behaviour and holds no Qt, so
the part worth testing can be tested without a window. ``MarkerDialog``
is a shell over it.
"""

import re

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class MarkerSelection:
    """A filtered, partly selected list of marker names.

    The filter only filters. Selecting is always the individual toggles,
    and :meth:`header_state` is computed from the filter and the
    selection rather than stored, which is what lets it follow the filter
    without anything keeping the two in step.

    Parameters
    ----------
    names : list[str]
        The panel. Held sorted, ignoring case, so ``p53`` sits between
        ``Pax5`` and ``Ki67`` by name rather than after every capital.
    """

    def __init__(self, names: list[str]):
        self._names = sorted(names, key=str.lower)
        self.selected: set[str] = set()
        self._pattern = ""
        self._matcher: re.Pattern | None = re.compile("")

    @property
    def pattern(self) -> str:
        return self._pattern

    @pattern.setter
    def pattern(self, value: str) -> None:
        self._pattern = value
        try:
            self._matcher = re.compile(value, re.IGNORECASE)
        except re.error:
            # A filter is typed one character at a time, so `[` on the
            # way to `[CD]` is normal rather than a mistake to report.
            self._matcher = None

    @property
    def pattern_is_valid(self) -> bool:
        return self._matcher is not None

    def visible(self) -> list[str]:
        """The names the filter leaves on screen, in panel order."""
        if self._matcher is None:
            return []
        return [name for name in self._names if self._matcher.search(name)]

    def toggle(self, name: str) -> None:
        self.selected.symmetric_difference_update({name})

    def header_state(self) -> str:
        """``"all"``, ``"none"`` or ``"partial"``, over the visible names.

        Nothing visible reads as ``"none"``: there is nothing on screen
        for the box to claim is selected.
        """
        visible = set(self.visible())
        if not visible:
            return "none"
        chosen = visible & self.selected
        if chosen == visible:
            return "all"
        return "partial" if chosen else "none"

    def toggle_all(self) -> None:
        """Select every visible name, or clear them if they all are.

        Partial goes to selected. Either direction discards the partial
        state, so neither is the safe one; going up matches reading the
        box as "is everything shown selected?" and answering "make it
        so".
        """
        visible = set(self.visible())
        if not visible:
            return
        if self.header_state() == "all":
            self.selected -= visible
        else:
            self.selected |= visible

    def chosen(self) -> list[str]:
        """The selected names in panel order, not in the order clicked."""
        return [name for name in self._names if name in self.selected]


class MarkerDialog(QDialog):
    """Ask which markers to put up.

    A shell over :class:`MarkerSelection`: every handler changes the
    selection and then redraws from it, so the widgets never hold state
    of their own.
    """

    def __init__(self, names: list[str], parent=None, title: str = "Choose markers"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.selection = MarkerSelection(names)

        self._search = QLineEdit()
        self._search.setPlaceholderText("filter, as a regular expression")
        self._search.textChanged.connect(self._on_search)

        self._header = QCheckBox("select all")
        self._header.setTristate(True)
        # `clicked` fires only for a real click, unlike `stateChanged`,
        # which also fires for the `setCheckState` in `_refresh`. That is
        # what keeps the redraw from being read back as a click.
        self._header.clicked.connect(self._on_header)

        self._count = QLabel()

        self._boxes: dict[str, QCheckBox] = {}
        markers = QVBoxLayout()
        for name in self.selection.visible():
            box = QCheckBox(name)
            box.clicked.connect(lambda _checked, n=name: self._on_marker(n))
            self._boxes[name] = box
            markers.addWidget(box)
        markers.addStretch(1)
        inner = QWidget()
        inner.setLayout(markers)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        top = QHBoxLayout()
        top.addWidget(self._header)
        top.addStretch(1)
        top.addWidget(self._count)

        layout = QVBoxLayout(self)
        layout.addWidget(self._search)
        layout.addLayout(top)
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        self.resize(360, 520)
        self._refresh()

    def chosen(self) -> list[str]:
        return self.selection.chosen()

    def _on_search(self, text: str) -> None:
        self.selection.pattern = text
        self._refresh()

    def _on_marker(self, name: str) -> None:
        self.selection.toggle(name)
        self._refresh()

    def _on_header(self, _checked: bool) -> None:
        # Qt has already moved the box; `_refresh` puts it where the
        # selection says it belongs, so a click can never leave it
        # showing "partial" as though that were an instruction.
        self.selection.toggle_all()
        self._refresh()

    def _refresh(self) -> None:
        visible = set(self.selection.visible())
        for name, box in self._boxes.items():
            box.setVisible(name in visible)
            box.setChecked(name in self.selection.selected)

        states = {
            "all": Qt.CheckState.Checked,
            "none": Qt.CheckState.Unchecked,
            "partial": Qt.CheckState.PartiallyChecked,
        }
        self._header.setCheckState(states[self.selection.header_state()])
        self._count.setText(
            f"{len(self.selection.selected)} of {len(self._boxes)} selected"
        )
        self._search.setStyleSheet(
            "" if self.selection.pattern_is_valid else "border: 1px solid #c0392b;"
        )


def ask_for_markers(names: list[str], parent=None, title: str = "Choose markers"):
    """Run :class:`MarkerDialog` and return the chosen names, or None.

    None means cancelled, which is different from choosing nothing.
    """
    dialog = MarkerDialog(names, parent=parent, title=title)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.chosen()
