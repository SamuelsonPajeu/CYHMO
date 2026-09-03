class CyhmoError(Exception):
    """Base de todos os erros previstos do mod (mensagens legíveis pelo usuário)."""


class ConfigError(CyhmoError):
    """Configuração ausente, inválida ou com chave desconhecida."""


class LanguagePackError(CyhmoError):
    """Pacote de idioma ausente ou malformado."""


class AudioDeviceError(CyhmoError):
    """Dispositivo de captura não encontrado ou inutilizável."""


class GrammarUnavailableError(CyhmoError):
    """Nenhuma gramática legível na memória do jogo."""


class InjectionError(CyhmoError):
    """Falha ao escrever a receita de injeção na memória do jogo."""


class UiServerError(CyhmoError):
    """A interface local não conseguiu abrir a porta configurada."""


class UpdateError(CyhmoError):
    """Não foi possível consultar, baixar ou aplicar a nova versão do mod."""


class LlmUnavailableError(CyhmoError):
    """O assistente não respondeu a tempo, falhou ou devolveu algo inutilizável.

    Distinto de recusa: recusar é decisão do assistente e continua valendo; falhar é
    problema dele, e não pode custar o comando do jogador."""
