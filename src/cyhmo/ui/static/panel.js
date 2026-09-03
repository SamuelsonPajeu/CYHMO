/* Painel: o que o mod está ouvindo agora e os últimos comandos. */

'use strict';

import { $, h, pill, replace, show, statusLabel, STATUS_PILL } from './dom.js';
import { t } from './i18n.js';

const RECENT_LIMIT = 5;
const BAR_WEIGHTS = [0.45, 0.75, 1, 0.75, 0.45];
const BAR_MIN_PX = 10;
const BAR_RANGE_PX = 42;
const LEVEL_GAIN = 3.2;

const RESULT_BY_STATUS = {
  injected: 'voice.sent',
  no_command: 'voice.notUnderstood',
  blocked_cannot_talk: 'voice.blocked',
  dry_run: 'voice.simulated',
  error: 'voice.failed',
  cancelled: 'voice.empty',
};

export function renderPanel(state) {
  const stage = voiceStage(state);
  const root = $('voice');
  root.className = `voice voice--${stage}`;
  $('voice-caption').textContent = captionFor(stage);
  renderTranscript(state, stage);
  renderCommands(state, stage);
  $('voice-result').textContent = resultFor(state, stage);
  replace($('voice-hint'), hintNodes(state));
  renderRecent(state);
  updateVoiceLevel(state);
}

/* O nível chega ~50x/s: mexe só nos próprios nós, nunca agenda render. */
export function updateVoiceLevel(state) {
  const root = $('voice');
  if (!root) return;
  const listening = root.classList.contains('voice--listening');
  const level = (state && state.level) || { rms: 0 };
  const amount = listening ? Math.min(1, Number(level.rms || 0) * LEVEL_GAIN) : 0;
  root.style.setProperty('--voice-level', amount.toFixed(3));
  const bars = root.querySelectorAll('.voice__bar');
  bars.forEach((bar, index) => {
    const weight = BAR_WEIGHTS[index % BAR_WEIGHTS.length];
    bar.style.height = `${BAR_MIN_PX + amount * weight * BAR_RANGE_PX}px`;
  });
}

function voiceStage(state) {
  if (!state.listening) return 'idle';
  if (state.phase === 'listening') return 'listening';
  if (state.phase === 'transcribing' || state.phase === 'interpreting') return 'thinking';
  if (state.phase === 'injecting') return 'sending';
  const last = state.last_utterance;
  if (!last || !last.finished) return 'idle';
  return last.status === 'injected' || last.status === 'dry_run' ? 'done' : 'error';
}

function captionFor(stage) {
  if (stage === 'listening') return t('voice.listening');
  if (stage === 'thinking') return t('voice.thinking');
  if (stage === 'sending') return t('voice.sending');
  return '';
}

/* Enquanto a fala está em curso o texto ainda não existe: a legenda já diz o
   estado, então repetir "Ouvindo…" aqui seria a mesma informação duas vezes. */
function renderTranscript(state, stage) {
  const node = $('voice-text');
  const last = state.last_utterance;
  const heard = last && last.transcript ? last.transcript.text : '';
  const settled = stage === 'done' || stage === 'error' || stage === 'sending';
  const text = settled && heard ? heard : '';
  const fallback = settled ? t('voice.empty') : stage === 'idle' ? t('voice.waiting') : '…';
  node.classList.toggle('voice__text--placeholder', !text);
  node.textContent = text || fallback;
}

function renderCommands(state, stage) {
  const last = state.last_utterance;
  const settled = stage === 'done' || stage === 'error' || stage === 'sending';
  const commands = settled && last && last.interpretation ? last.interpretation.commands || [] : [];
  replace($('voice-commands'), commands.map((command) => pill(command.key, 'pill--injecting', 'pill--lg')));
}

function resultFor(state, stage) {
  if (stage !== 'done' && stage !== 'error') return '';
  const last = state.last_utterance;
  if (!last) return '';
  const injection = last.injection && last.injection.result;
  if (last.status === 'injected' && injection && injection.matched === false) return t('voice.ignored');
  return t(RESULT_BY_STATUS[last.status] || 'voice.notUnderstood');
}

/* A tecla vira <kbd> no meio da frase, então o texto do locale é dividido em vez
   de interpolado — nenhum locale precisa carregar HTML. */
function hintNodes(state) {
  if (!state.listening) return [h('span', { text: t('voice.paused') })];
  const activation = state.config.activation || {};
  if (activation.mode === 'vad') return [h('span', { text: t('voice.hintHandsFree') })];
  const key = activation.ptt_hotkey;
  if (!key) return [h('span', { text: t('voice.hintNoKey') })];
  const [before, after] = t('voice.hint').split('{key}');
  return [h('span', { text: before }), h('kbd', { text: key }), h('span', { text: after || '' })];
}

function renderRecent(state) {
  const rows = (state.history || []).slice(0, RECENT_LIMIT);
  show($('recent-empty'), rows.length === 0);
  replace(
    $('recent-rows'),
    rows.map((row) =>
      h('tr', {}, [
        h('td', {}, [pill(statusLabel(row.status), STATUS_PILL[row.status])]),
        h('td', { class: 'td--wrap', text: row.text || t('event.empty') }),
        h('td', { class: 'td--wrap', text: (row.keys || []).join(', ') || '—' }),
      ]),
    ),
  );
}
