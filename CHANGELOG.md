# Changelog

A seção de cada versão é publicada no corpo da release e aparece na janela
"Nova versão disponível" dentro do mod. A janela corta as notas em 1200
caracteres (`NOTES_LIMIT` em `src/cyhmo/update/release.py`), então cada seção
é escrita para caber inteira nesse limite.

Versões anteriores à 1.0.2 não têm seção aqui; suas notas estão nas releases
do GitHub.

## 1.0.3

- Comando longo não trava mais a partida. A senha da Sun Suite tem 69
  caracteres e o mod recusava qualquer comando acima de 63, sem saída
  possível; o teto agora é 127 e ela é enviada inteira.
- "Suíte do sol", "cama", "penteadeira", "sala" e outros nomes de cômodos e
  objetos passam a ser entendidos em português. Sem eles alguns lugares não
  tinham como ser alcançados falando — inclusive a própria Sun Suite, que
  prendia a história.
- Verbo sozinho não é mais dado como enviado. O jogo aceita "procura" e "vai"
  sem alvo e não faz nada com eles, então o mod passa a juntar o verbo ao
  objeto da cena; sem objeto claro, avisa que falta dizer o alvo em vez de
  parecer que deu certo.
- "Atrás", "embaixo", "do lado" e "dentro" agora contam na escolha: entre dois
  comandos quase iguais ganha o que combina com o lugar que você disse.
- O assistente deixa de perder resposta certa. Quando ele acerta o comando e
  continua falando, o comando é aproveitado; quando responde duas coisas
  diferentes, nada é enviado.

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
