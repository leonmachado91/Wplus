"""Speaker mapper — mantém mapa de labels pyannote → nomes customizados por sessão."""

from __future__ import annotations

from typing import Optional

# Paleta Gruvbox para speakers (cycling)
SPEAKER_COLORS = [
    "#83a598",  # blue
    "#fe8019",  # orange
    "#b8bb26",  # green
    "#fabd2f",  # yellow
    "#d3869b",  # purple
    "#8ec07c",  # aqua
    "#fb4934",  # red
    "#ebdbb2",  # fg (fallback)
]


class SpeakerMapper:
    """Traduz labels pyannote (SPEAKER_00, etc.) para nomes amigáveis com cores consistentes."""

    def __init__(self, custom_names: dict[str, str] | None = None):
        # Ex: {"SPEAKER_00": "Leandro", "SPEAKER_01": "DM"}
        self._custom: dict[str, str] = custom_names or {}
        # Cache de cores por speaker
        self._colors: dict[str, str] = {}
        self._index: int = 0

    def update_custom_names(self, names: dict[str, str]) -> None:
        self._custom = names

    def display_name(self, speaker_label: str) -> str:
        """Retorna nome amigável: custom name se configurado, senão label original."""
        return self._custom.get(speaker_label, speaker_label)

    def color(self, speaker_label: str) -> str:
        """Retorna cor HTML consistente para o speaker."""
        if speaker_label not in self._colors:
            self._colors[speaker_label] = SPEAKER_COLORS[self._index % len(SPEAKER_COLORS)]
            self._index += 1
        return self._colors[speaker_label]

    def reset(self) -> None:
        """Limpa estado da sessão (chamar ao iniciar nova gravação)."""
        self._colors.clear()
        self._index = 0
