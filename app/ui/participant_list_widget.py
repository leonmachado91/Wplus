"""Table widget showing connected participants in the multi-device panel."""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

logger = logging.getLogger(__name__)


class ParticipantListWidget(QWidget):
    """Displays live participant table with mute / rename / kick actions."""

    def __init__(
        self,
        on_mute: Optional[Callable[[str, bool], None]] = None,
        on_rename: Optional[Callable[[str, str], None]] = None,
        on_kick: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_mute = on_mute
        self._on_rename = on_rename
        self._on_kick = on_kick
        self._tokens: list[str] = []  # ordered list of tokens for row lookup

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Nome", "Modo", "Chunks", "Ações"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(
            "QTableWidget { border: none; background: #282828; }"
            "QTableWidget::item { padding: 6px 8px; }"
            "QHeaderView::section { background: #3c3836; color: #a89984; border: none; padding: 4px 8px; }"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def update_participants(self, participants: list) -> None:
        """Re-render the table from a list of Participant objects."""
        current_tokens = {p.token for p in participants}

        # Remove rows for participants who left
        rows_to_remove = [
            i for i, t in enumerate(self._tokens) if t not in current_tokens
        ]
        for row in reversed(rows_to_remove):
            self._table.removeRow(row)
            self._tokens.pop(row)

        for participant in participants:
            token = participant.token
            if token in self._tokens:
                row = self._tokens.index(token)
                self._update_row(row, participant)
            else:
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._tokens.append(token)
                self._update_row(row, participant)
                self._set_action_buttons(row, token, participant.display_name)

    def clear(self) -> None:
        self._table.setRowCount(0)
        self._tokens.clear()

    def _update_row(self, row: int, participant) -> None:
        name_item = QTableWidgetItem(participant.display_name)
        name_color = Qt.GlobalColor.yellow if participant.muted else Qt.GlobalColor.white
        name_item.setForeground(name_color)
        self._table.setItem(row, 0, name_item)
        self._table.setItem(row, 1, QTableWidgetItem(participant.mode))
        self._table.setItem(row, 2, QTableWidgetItem(str(participant.chunk_count)))

    def _set_action_buttons(self, row: int, token: str, display_name: str) -> None:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        btn_mute = QPushButton("Mute")
        btn_mute.setFixedWidth(52)
        btn_mute.setStyleSheet(
            "QPushButton { background: #504945; color: #ebdbb2; border-radius: 4px; padding: 3px 6px; font-size: 11px; }"
            "QPushButton:hover { background: #665c54; }"
        )
        btn_mute.clicked.connect(lambda _, t=token: self._toggle_mute(t, btn_mute))
        layout.addWidget(btn_mute)

        btn_kick = QPushButton("Kick")
        btn_kick.setFixedWidth(44)
        btn_kick.setStyleSheet(
            "QPushButton { background: #cc241d; color: #ebdbb2; border-radius: 4px; padding: 3px 6px; font-size: 11px; }"
            "QPushButton:hover { background: #fb4934; }"
        )
        btn_kick.clicked.connect(lambda _, t=token: self._kick(t))
        layout.addWidget(btn_kick)

        self._table.setCellWidget(row, 3, container)

    def _toggle_mute(self, token: str, btn: QPushButton) -> None:
        if token not in self._tokens:
            return
        row = self._tokens.index(token)
        name_item = self._table.item(row, 0)
        currently_muted = name_item and name_item.foreground().color() == Qt.GlobalColor.yellow
        new_muted = not currently_muted
        if name_item:
            name_item.setForeground(Qt.GlobalColor.yellow if new_muted else Qt.GlobalColor.white)
        btn.setText("Unmute" if new_muted else "Mute")
        if self._on_mute:
            self._on_mute(token, new_muted)

    def _kick(self, token: str) -> None:
        if self._on_kick:
            self._on_kick(token)
