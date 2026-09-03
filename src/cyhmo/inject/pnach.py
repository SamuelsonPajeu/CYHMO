"""Geração do arquivo ``.pnach`` do patch do portão do pad.

Alternativa ao modo ``runtime``: em vez de o mod aplicar e restaurar a instrução a
cada injeção, o PCSX2 a mantém aplicada enquanto o jogo roda. Texto simples e
auditável — nada do jogo vai dentro, só um endereço e um valor.
"""

from __future__ import annotations

from pathlib import Path

from cyhmo.inject.recipe import WriteRecipe

PATCH_GROUP = "CYHMO"
PLACE_EVERY_FRAME = 1
CPU_EE = "EE"
TYPE_WORD = "word"


def pnach_filename(recipe: WriteRecipe) -> str:
    """O PCSX2 exige ``SERIAL_CRC.pnach`` — só o CRC não é encontrado."""
    return f"{recipe.serial}_{recipe.crc}.pnach"


def render_pnach(recipe: WriteRecipe, place: int = PLACE_EVERY_FRAME) -> str:
    patch = recipe.ascii_words.code_patch
    return "\n".join(
        (
            f"gametitle=Lifeline ({recipe.serial})",
            f"comment=Gerado por cyhmo a partir de {pnach_filename(recipe)[:-6]}; "
            "sem este patch o jogo ignora o comando injetado.",
            "",
            f"[{PATCH_GROUP}]",
            "author=cyhmo",
            "description=Neutraliza a leitura do pad que bloqueia o resultado do reconhecimento, "
            "para que o comando escrito pelo mod seja aceito. Uma instrucao, apenas em memoria.",
            f"// 0x{patch.addr:08X}: 0x{patch.original:08X} -> 0x{patch.patched:08X}",
            f"patch={place},{CPU_EE},{patch.addr:08X},{TYPE_WORD},{patch.patched:08X}",
            "",
        )
    )


def write_pnach(recipe: WriteRecipe, directory: Path | str, place: int = PLACE_EVERY_FRAME) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / pnach_filename(recipe)
    target.write_text(render_pnach(recipe, place), encoding="ascii", newline="\n")
    return target
