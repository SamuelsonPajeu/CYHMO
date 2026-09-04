# Changelog

A seção de cada versão é publicada no corpo da release e aparece na janela
"Nova versão disponível" dentro do mod. A janela corta as notas em 1200
caracteres (`NOTES_LIMIT` em `src/cyhmo/update/release.py`), então cada seção
é escrita para caber inteira nesse limite.

Versões anteriores à 1.0.2 não têm seção aqui; suas notas estão nas releases
do GitHub.

## 1.0.2

- Escolha do modelo de reconhecimento pela interface, com download conferido
  pelo SHA-1 que o whisper.cpp publica. Trocar e remover sem sair do mod.
- Suporte a GPU NVIDIA (cuBLAS): o mod confere placa e driver CUDA antes de
  oferecer a opção e baixa o build sob demanda. Em AMD ou Intel a opção não
  aparece, porque o whisper.cpp não publica build de GPU para elas.
- 12 idiomas novos: alemão, árabe, bengali, coreano, francês, hindi, indonésio,
  italiano, japonês, russo, turco e vietnamita. São 16 no total.
- A primeira execução abre no idioma do Windows, com inglês de reserva.
- A atualização automática agora confere o SHA-256 do pacote contra o arquivo
  publicado na release. Pacote que não bate não é instalado.
- A interface local só aceita requisição da própria máquina.
