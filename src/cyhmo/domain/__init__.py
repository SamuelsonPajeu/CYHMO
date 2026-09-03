"""Model da arquitetura (MVVM): contratos canônicos, eventos e ports.

Nenhum módulo daqui importa infraestrutura (áudio, modelos, sockets, UI).
"""

from cyhmo.domain.contracts import (
    AudioSegment,
    Candidate,
    CommandRef,
    GameState,
    InjectResult,
    Interpretation,
    Transcript,
    Utterance,
)
from cyhmo.domain.errors import (
    ConfigError,
    LanguagePackError,
    CyhmoError,
)
from cyhmo.domain.language_pack import LanguagePack

__all__ = [
    "AudioSegment",
    "Candidate",
    "CommandRef",
    "ConfigError",
    "GameState",
    "InjectResult",
    "Interpretation",
    "LanguagePack",
    "LanguagePackError",
    "CyhmoError",
    "Transcript",
    "Utterance",
]
