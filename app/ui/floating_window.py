"""Floating button window — always-on-top circular button for voice dictation."""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import (
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    pyqtSignal,
    pyqtProperty,  # type: ignore[attr-defined]
)
from PyQt6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

# Gruvbox palette
_COLOR_IDLE      = QColor("#83a598")   # teal (idle)
_COLOR_RECORDING = QColor("#fb4934")   # red  (recording)
_COLOR_BG        = QColor("#282828")   # background sólido da janela
_COLOR_BG_WIDGET = QColor("#1d2021")   # background mais escuro do círculo
_COLOR_TEXT      = QColor("#a89984")   # label muted
_COLOR_BORDER    = QColor("#3c3836")   # borda da janela

_BUTTON_SIZE  = 68
_BORDER_WIDTH = 3
_LABEL_HEIGHT = 18
_CLOSE_BTN_H  = 24            # ↑ aumentado de 18 → 24
_WINDOW_W     = 100           # ↑ um pouco mais largo
_WINDOW_H     = _CLOSE_BTN_H + _BUTTON_SIZE + _LABEL_HEIGHT + 16
_CORNER_RADIUS = 12


class FloatingWindow(QWidget):
    """Frameless, always-on-top circular button for the Floating Button mode.

    Qt.Tool flag prevents focus theft — clicking this button does NOT
    transfer keyboard focus away from the active text field.

    Signals:
        closed:             Emitted when the user closes the window (X or close()).
        recording_toggled:  Emitted with True/False when the button is clicked.
    """

    closed = pyqtSignal()
    recording_toggled = pyqtSignal(bool)  # True = start, False = stop

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool,
        )
        # NÃO usa WA_TranslucentBackground → fundo sólido, sem click-through
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(_WINDOW_W, _WINDOW_H)

        self._is_recording = False
        self._border_color = _COLOR_IDLE
        self._border_opacity: float = 1.0

        # drag state
        self._drag_pos: Optional[QPoint] = None
        self._drag_start: Optional[QPoint] = None

        # pulse animation when recording
        self._anim = QPropertyAnimation(self, b"borderOpacity")
        self._anim.setDuration(700)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.3)
        self._anim.setLoopCount(-1)

        self._build_ui()
        self._position_top_right()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 6)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # ── top bar: close button ────────────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(4, 0, 4, 0)
        top_bar.addStretch()

        self._btn_close = QPushButton("✕")
        self._btn_close.setFixedSize(22, 22)          # ↑ 16×16 → 22×22
        self._btn_close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_close.setStyleSheet(
            "QPushButton {"
            "  background: #3c3836;"
            "  color: #a89984;"
            "  font-size: 12px;"
            "  font-weight: bold;"
            "  border: none;"
            "  border-radius: 11px;"
            "  padding: 0;"
            "}"
            "QPushButton:hover { background: #fb4934; color: #fbf1c7; }"
        )
        self._btn_close.clicked.connect(self.close)
        top_bar.addWidget(self._btn_close)
        layout.addLayout(top_bar)

        # espaço para o círculo pintado
        self._circle_space = QWidget()
        self._circle_space.setFixedSize(_WINDOW_W, _BUTTON_SIZE + 4)
        self._circle_space.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self._circle_space)

        self._status_label = QLabel("Clique para gravar")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._status_label.setStyleSheet(
            f"color: {_COLOR_TEXT.name()}; font-size: 9px; background: transparent;"
        )
        self._status_label.setFixedHeight(_LABEL_HEIGHT)
        layout.addWidget(self._status_label)

    # ── background painting ───────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ── fundo sólido com cantos arredondados ────────────────────────
        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, _WINDOW_W, _WINDOW_H, _CORNER_RADIUS, _CORNER_RADIUS)
        painter.fillPath(bg_path, _COLOR_BG)

        # borda fina da janela
        painter.setPen(QPen(_COLOR_BORDER, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(bg_path)

        # ── círculo do botão ────────────────────────────────────────────
        cx = _WINDOW_W // 2
        cy = _CLOSE_BTN_H + (_BUTTON_SIZE + 4) // 2
        r  = _BUTTON_SIZE // 2

        # fundo do círculo
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_COLOR_BG_WIDGET)
        painter.drawEllipse(QPoint(cx, cy), r - _BORDER_WIDTH, r - _BORDER_WIDTH)

        # borda animada
        border_color = QColor(self._border_color)
        border_color.setAlphaF(self._border_opacity)
        pen = QPen(border_color, _BORDER_WIDTH)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        offset = _BORDER_WIDTH // 2
        painter.drawEllipse(QPoint(cx, cy), r - offset, r - offset)

        # ── ícone dentro do círculo ─────────────────────────────────────
        if self._is_recording:
            # ponto vermelho pulsante
            dot_color = QColor(_COLOR_RECORDING)
            dot_color.setAlphaF(self._border_opacity)
            painter.setBrush(dot_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(cx, cy), 11, 11)
        else:
            self._draw_mic(painter, cx, cy)

        painter.end()

    def _draw_mic(self, painter: QPainter, cx: int, cy: int) -> None:
        """Desenha um ícone de microfone limpo e proporcional."""
        color = _COLOR_IDLE
        pen = QPen(color, 2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # corpo do mic: retângulo arredondado
        mic_w, mic_h = 11, 17
        mic_x = cx - mic_w // 2
        mic_y = cy - 14
        painter.drawRoundedRect(mic_x, mic_y, mic_w, mic_h, 5.5, 5.5)

        # arco inferior (suporte)
        arc_w, arc_h = 20, 14
        arc_x = cx - arc_w // 2
        arc_y = cy - 2
        painter.drawArc(arc_x, arc_y, arc_w, arc_h, 0, -180 * 16)

        # haste
        painter.drawLine(cx, cy + arc_h // 2, cx, cy + arc_h // 2 + 5)
        # base
        base_half = 7
        painter.drawLine(cx - base_half, cy + arc_h // 2 + 5,
                         cx + base_half, cy + arc_h // 2 + 5)

    # ── positioning ───────────────────────────────────────────────────────

    def _position_top_right(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.move(40, 40)
            return
        geom = screen.availableGeometry()
        x = geom.right() - _WINDOW_W - 40
        y = geom.top() + 40
        self.move(x, y)

    def restore_position(self, x: int, y: int) -> None:
        """Move para uma posição previamente salva."""
        self.move(x, y)

    # ── recording state ───────────────────────────────────────────────────

    def set_recording(self, recording: bool) -> None:
        self._is_recording = recording
        if recording:
            self._border_color = _COLOR_RECORDING
            self._anim.start()
            self._status_label.setText("Gravando...")
            self._status_label.setStyleSheet(
                "color: #fb4934; font-size: 9px; background: transparent; font-weight: bold;"
            )
        else:
            self._anim.stop()
            self._border_opacity = 1.0
            self._border_color = _COLOR_IDLE
            self._status_label.setText("Clique para gravar")
            self._status_label.setStyleSheet(
                f"color: {_COLOR_TEXT.name()}; font-size: 9px; background: transparent;"
            )
        self.update()

    def flash_injected(self) -> None:
        """Mostra confirmação breve após texto ser injetado."""
        self._status_label.setText("Texto injetado!")
        self._status_label.setStyleSheet(
            "color: #b8bb26; font-size: 9px; background: transparent;"
        )
        QTimer.singleShot(
            1200,
            lambda: self._status_label.setText("Gravando...")
            if self._is_recording
            else self._status_label.setText("Clique para gravar"),
        )

    # ── animated property ─────────────────────────────────────────────────

    @pyqtProperty(float)
    def borderOpacity(self) -> float:  # noqa: N802
        return self._border_opacity

    @borderOpacity.setter  # type: ignore[no-redef]
    def borderOpacity(self, value: float) -> None:  # noqa: N802
        self._border_opacity = value
        self.update()

    # ── mouse events (drag + click) ───────────────────────────────────────

    def _in_circle(self, pos: QPoint) -> bool:
        cx = _WINDOW_W // 2
        cy = _CLOSE_BTN_H + (_BUTTON_SIZE + 4) // 2
        r  = _BUTTON_SIZE // 2
        dx = pos.x() - cx
        dy = pos.y() - cy
        return dx * dx + dy * dy <= r * r

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.globalPosition().toPoint()
            self._drag_pos   = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            delta = (event.globalPosition().toPoint() - self._drag_start).manhattanLength()
            if delta < 6 and self._in_circle(event.position().toPoint()):
                self._is_recording = not self._is_recording
                self.recording_toggled.emit(self._is_recording)
                self.set_recording(self._is_recording)
        self._drag_pos   = None
        self._drag_start = None

    # ── close ─────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802
        self.closed.emit()
        event.accept()
