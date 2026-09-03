/* Estado da View: o snapshot do servidor sempre manda; os eventos só o adiantam. */

'use strict';

import { api } from './api.js';
import { fmtHex } from './dom.js';
import { t } from './i18n.js';

const HISTORY_LIMIT = 50;
const CONSOLE_LIMIT = 200;

export const store = {
  state: null,
  activeTab: 'panel',
  consoleLevel: 'debug',
  selectedUtterance: null,
  settingsVersion: 0,
  renderedSettingsVersion: -1,
  socket: null,
  renderQueued: false,
  hotkeyRecording: false,
  lifecycle: null,
  consoleSeq: 0,
  renderedConsole: null,
  socketDown: false,
  stopped: false,
};

let renderer = () => {};

export function setRenderer(callback) {
  renderer = callback;
}

export function scheduleRender() {
  if (store.renderQueued) return;
  store.renderQueued = true;
  requestAnimationFrame(() => {
    store.renderQueued = false;
    renderer();
  });
}

export function isAdvanced() {
  return Boolean(store.state) && store.state.mode === 'advanced';
}

export function isGrammarStale(state) {
  /* A perda da cena vem do GameState vivo. `grammar.stale` só é escrito pelo evento de
     troca de gramática, que a camada 3 publica apenas quando ADOTA uma — nunca quando
     perde a que tinha; ler dali deixava este estado inalcançável. */
  return Boolean((state.state || {}).grammar_stale);
}

export function totalMs(latency) {
  if (latency.total !== undefined) return latency.total;
  return Object.values(latency).reduce((sum, value) => sum + Number(value || 0), 0);
}

export async function refreshState() {
  applySnapshot(await api('/api/state'));
}

export function applySnapshot(snapshot) {
  store.state = snapshot;
  (snapshot.console || []).forEach((line) => {
    line.seq = ++store.consoleSeq;
  });
  store.renderedConsole = null;
  if (!store.hotkeyRecording) store.settingsVersion += 1;
  scheduleRender();
}

function ensureUtterance(state, uttId, time) {
  const current = state.last_utterance;
  if (current && current.utt_id === uttId && !current.finished) return current;
  const record = {
    utt_id: uttId,
    t_start: time,
    source: 'mic',
    duration_ms: 0,
    wav_path: null,
    transcript: null,
    interpretation: null,
    state: null,
    injection: null,
    latency_ms: {},
    total_ms: 0,
    status: null,
    error: null,
    finished: false,
  };
  state.last_utterance = record;
  state.utt_id = uttId;
  return record;
}

function historyRow(record, budget) {
  const interpretation = record.interpretation || {};
  const total = totalMs(record.latency_ms);
  return {
    utt_id: record.utt_id,
    t: record.t_start,
    status: record.status,
    text: record.transcript ? record.transcript.text : '',
    keys: (interpretation.commands || []).map((command) => command.key),
    method: interpretation.method || 'none',
    confidence: interpretation.confidence || 0,
    total_ms: total,
    over_budget: budget.total !== undefined ? total > budget.total : false,
    detail: record,
  };
}

function summarizeEvent(event) {
  const utt = event.utt_id;
  switch (event.kind) {
    case 'phase':
      return utt ? t('event.phaseUtt', { phase: event.phase, utt }) : t('event.phase', { phase: event.phase });
    case 'utterance':
      return t('event.utterance', { utt, duration: Number(event.duration_ms).toFixed(0), source: event.source });
    case 'transcript':
      return event.transcript
        ? t('event.transcript', {
            utt,
            text: event.transcript.text,
            lang: event.transcript.lang,
            confidence: Number(event.transcript.confidence).toFixed(2),
            latency: Number(event.latency_ms).toFixed(0),
          })
        : t('event.transcriptEmpty', { utt });
    case 'interpretation': {
      const interpretation = event.interpretation;
      if (!interpretation) return t('event.interpretationNone', { utt });
      return t('event.interpretation', {
        utt,
        keys: interpretation.commands.map((command) => command.key).join(', ') || t('event.none'),
        method: interpretation.method,
        confidence: Number(interpretation.confidence).toFixed(2),
      });
    }
    case 'injection': {
      const keys = (event.commands || []).map((command) => command.key).join(', ') || t('event.empty');
      if (!event.result) return t('event.injectionNoResult', { utt, keys });
      const verdict = event.result.ok ? t('event.injectionOk') : t('event.injectionFailed', { error: event.result.error });
      return t('event.injection', { utt, keys, verdict, latency: Number(event.result.latency_ms).toFixed(0) });
    }
    case 'finished':
      return t('event.finished', { utt, status: event.status, total: totalMs(event.latency_ms || {}).toFixed(0) });
    case 'grammar':
      return t(event.stale ? 'event.grammarStale' : 'event.grammar', {
        size: event.size,
        pointer: fmtHex(event.pointer_new),
        fresh: event.new_in_session,
      });
    case 'state':
      return event.state
        ? t('event.state', { mode: event.state.mode, canTalk: event.state.can_talk })
        : t('event.stateUnknown');
    case 'component':
      return t('event.component', { component: event.component, status: event.status });
    case 'config':
      return t(event.restart_required ? 'event.configRestart' : 'event.config', {
        sections: (event.sections || []).join(', '),
      });
    default:
      return event.kind;
  }
}

function consoleLine(event) {
  if (event.kind === 'log') {
    return { t: event.t, level: event.level, source: event.source, message: event.message };
  }
  return { t: event.t, level: 'debug', source: event.kind, message: summarizeEvent(event) };
}

export function applyEvent(event) {
  const state = store.state;
  if (!state) return 'ignored';
  switch (event.kind) {
    /* A volta a idle carrega o utt_id recém-encerrado: criar registro aqui
       apagaria o resultado do Painel no instante em que ele aparece. */
    case 'phase':
      state.phase = event.phase;
      if (event.utt_id && event.phase !== 'idle') ensureUtterance(state, event.utt_id, event.t);
      break;
    case 'utterance': {
      const record = ensureUtterance(state, event.utt_id, event.t);
      record.duration_ms = event.duration_ms;
      record.source = event.source;
      record.wav_path = event.wav_path;
      break;
    }
    case 'transcript': {
      const record = ensureUtterance(state, event.utt_id, event.t);
      record.transcript = event.transcript;
      record.latency_ms.stt = event.latency_ms;
      break;
    }
    case 'interpretation': {
      const record = ensureUtterance(state, event.utt_id, event.t);
      if (event.interpretation) {
        record.interpretation = event.interpretation;
        record.latency_ms.intent = event.interpretation.latency_ms;
      }
      if (event.state) record.state = event.state;
      break;
    }
    case 'injection': {
      const record = ensureUtterance(state, event.utt_id, event.t);
      record.injection = { commands: event.commands, result: event.result };
      if (event.result) record.latency_ms.inject = event.result.latency_ms;
      break;
    }
    case 'finished': {
      const record = ensureUtterance(state, event.utt_id, event.t);
      Object.assign(record.latency_ms, event.latency_ms || {});
      record.status = event.status;
      record.error = event.error;
      record.finished = true;
      record.total_ms = totalMs(record.latency_ms);
      state.history.unshift(historyRow(record, state.budget_ms || {}));
      state.history.length = Math.min(state.history.length, HISTORY_LIMIT);
      state.utt_id = null;
      break;
    }
    case 'grammar':
      state.grammar = Object.assign({}, state.grammar, {
        size: event.size,
        stale: event.stale,
        pointer: event.pointer_new,
        blob_address: event.blob_address,
        changed_at: event.t,
        version: (state.grammar.version || 0) + 1,
        new_in_session: event.new_in_session,
      });
      break;
    case 'state':
      if (event.state) state.state = event.state;
      break;
    case 'component':
      state.components[event.component] = { status: event.status, detail: event.detail };
      break;
    case 'level':
      state.level = { rms: event.rms, peak: event.peak };
      return 'level';
    case 'config':
      refreshState();
      break;
    case 'dropped':
      state.dropped_events = event.count;
      refreshState();
      return 'ignored';
    default:
      break;
  }
  state.console.push(Object.assign(consoleLine(event), { seq: ++store.consoleSeq }));
  if (state.console.length > CONSOLE_LIMIT) state.console.splice(0, state.console.length - CONSOLE_LIMIT);
  return event.kind;
}

export function getPath(tree, dotted) {
  return dotted.split('.').reduce((node, key) => (node === undefined || node === null ? undefined : node[key]), tree);
}

export function setPath(tree, dotted, value) {
  const keys = dotted.split('.');
  let node = tree;
  keys.slice(0, -1).forEach((key) => {
    if (typeof node[key] !== 'object' || node[key] === null) node[key] = {};
    node = node[key];
  });
  node[keys[keys.length - 1]] = value;
}
