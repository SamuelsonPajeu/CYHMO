/* Ferramentas (modo avançado): estado, latência, testes e calibração. */

'use strict';

import { api } from './api.js';
import {
  $,
  empty,
  fmtMs,
  fmtPct,
  fmtScore,
  h,
  kvList,
  levelMeter,
  metric,
  pill,
  preJson,
  progress,
  replace,
} from './dom.js';
import { candidateMetrics, latencyMetrics } from './history.js';
import { t } from './i18n.js';
import { store } from './store.js';
import { toast } from './toast.js';

const CALIBRATION_POLL_MS = 1500;
const CLIPPING = 0.99;
const TOO_QUIET = 0.01;

let calibrationTimer = null;

export function renderTools(state) {
  renderStatus(state);
  replace($('latency-body'), [
    state.last_utterance ? null : h('p', { class: 'text-muted text-sm', text: t('tools.latency.waiting') }),
    ...latencyMetrics(state, state.last_utterance),
  ]);
}

function renderStatus(state) {
  const game = state.state || {};
  replace($('status-body'), [
    kvList([
      [t('tools.status.session'), state.session_id],
      [t('tools.status.listening'), t(state.listening ? 'common.on' : 'common.off')],
    ]),
    state.dropped_events
      ? h('div', { class: 'row' }, [pill(t('tools.status.dropped', { count: state.dropped_events }), 'pill--warning')])
      : null,
    levelMeter(t('tools.status.level')),
    kvList([
      [t('tools.status.mode'), game.mode || t('common.unknown')],
      [t('tools.status.canTalk'), canTalkLabel(game.can_talk)],
      [t('tools.status.room'), game.room],
      [t('tools.status.hp'), game.hp],
      [t('tools.status.enemies'), game.enemies],
      [t('tools.status.grammar'), grammarLabel(game)],
    ]),
  ]);
}

function canTalkLabel(value) {
  if (value === null || value === undefined) return t('tools.status.canTalkUnmapped');
  return t(value ? 'common.yes' : 'common.no');
}

function grammarLabel(game) {
  if (game.grammar_size === undefined) return null;
  return game.grammar_stale ? t('tools.status.grammarStale', { size: game.grammar_size }) : String(game.grammar_size);
}

export function bindTools() {
  bindInterpret();
  bindInject();
  bindMicTest();
  bindInjector();
  bindCalibration();
}

function bindInterpret() {
  $('interpret-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = $('interpret-text').value.trim();
    if (!text) return;
    const target = $('interpret-result');
    replace(target, h('p', { class: 'text-muted text-sm', text: t('tools.interpret.running') }));
    try {
      replace(target, interpretationView(await api('/api/interpret', { method: 'POST', body: { text } })));
    } catch (error) {
      replace(target, h('p', { class: 'text-danger text-sm', text: error.message }));
    }
  });
}

function bindInject() {
  $('inject-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = $('inject-text').value.trim();
    if (!text) return;
    const target = $('inject-result');
    replace(target, h('p', { class: 'text-muted text-sm', text: t('tools.inject.running') }));
    try {
      replace(target, injectResultView(await api('/api/inject', { method: 'POST', body: { text } })));
    } catch (error) {
      replace(target, h('p', { class: 'text-danger text-sm', text: error.message }));
    }
  });
}

function bindMicTest() {
  $('mic-test-button').addEventListener('click', async () => {
    const button = $('mic-test-button');
    const target = $('mic-test-result');
    button.disabled = true;
    replace(target, h('p', { class: 'text-muted text-sm', text: t('settings.mic.testing') }));
    try {
      const result = await api('/api/mic-test', { method: 'POST', body: { seconds: 2 } });
      replace(target, [
        metric(t('tools.mic.peak'), fmtPct(result.peak), Number(result.peak || 0), {
          variant: 'accent',
          over: result.peak > CLIPPING,
          help: result.peak > CLIPPING ? t('settings.mic.testClipping') : null,
        }),
        metric(t('tools.mic.rms'), fmtPct(result.rms), Number(result.rms || 0), {
          variant: 'success',
          help: result.rms < TOO_QUIET ? t('settings.mic.testLow') : null,
        }),
      ]);
    } catch (error) {
      replace(target, h('p', { class: 'text-danger text-sm', text: error.message }));
    } finally {
      button.disabled = false;
    }
  });
}

function bindInjector() {
  replace($('injector-status'), empty(t('tools.injector.empty')));
  $('injector-refresh').addEventListener('click', async () => {
    const target = $('injector-status');
    try {
      replace(target, kvList(Object.entries(await api('/api/injector')), { mono: true }));
    } catch (error) {
      replace(target, h('p', { class: 'text-danger text-sm', text: error.message }));
    }
  });
}

function interpretationView(interpretation) {
  const commands = interpretation.commands || [];
  return h('div', { class: 'stack' }, [
    h(
      'div',
      { class: 'row' },
      commands.length
        ? commands.map((command) => pill(command.key, 'pill--listening', 'pill--lg'))
        : [pill(t('history.noCommand'), 'pill--muted', 'pill--lg')],
    ),
    kvList([
      [t('history.method'), t(`method.${interpretation.method}`)],
      [t('history.confidence'), fmtScore(interpretation.confidence)],
      [t('history.reason'), interpretation.reason || '—'],
      [t('history.normalized'), interpretation.normalized_text || '—'],
      [t('history.latency'), fmtMs(interpretation.latency_ms)],
    ]),
    ...candidateMetrics(interpretation, commands),
  ]);
}

function injectResultView(result) {
  return h('div', { class: 'stack' }, [
    h('div', { class: 'row' }, [
      pill(result.ok ? t('history.writeOk') : t('voice.failed'), result.ok ? 'pill--ok' : 'pill--error'),
      pill(oracleText(result.matched), result.matched ? 'pill--injecting' : 'pill--warning'),
      h('span', { class: 'mono text-muted', text: fmtMs(result.latency_ms) }),
    ]),
    result.error ? h('p', { class: 'text-danger text-sm', text: result.error }) : null,
    result.payload_echo && Object.keys(result.payload_echo).length ? preJson(result.payload_echo) : null,
  ]);
}

function oracleText(matched) {
  if (matched === null || matched === undefined) return t('history.oracleUnknown');
  return matched ? t('history.oracleMatched') : t('history.oracleMissed');
}

function bindCalibration() {
  $('calibration-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      showCalibration(
        await api('/api/calibration/start', {
          method: 'POST',
          body: {
            dataset: $('calibration-dataset').value.trim(),
            spontaneous: $('calibration-spontaneous').value.trim(),
            grid: $('calibration-grid').value.trim(),
          },
        }),
      );
    } catch (error) {
      toast(t('tools.calibration.failed', { error: error.message }), 'error');
    }
  });
  $('calibration-cancel').addEventListener('click', async () => {
    showCalibration(await api('/api/calibration/cancel', { method: 'POST' }));
  });
}

export async function refreshCalibration() {
  try {
    showCalibration(await api('/api/calibration'));
  } catch (error) {
    replace($('calibration-result'), h('p', { class: 'text-danger text-sm', text: error.message }));
  }
}

function showCalibration(status) {
  fillDefault($('calibration-dataset'), status.dataset);
  fillDefault($('calibration-spontaneous'), status.spontaneous);
  fillDefault($('calibration-grid'), status.grid);
  $('calibration-start').disabled = Boolean(status.running);
  $('calibration-cancel').hidden = !status.running;
  scheduleCalibrationPoll(status.running);
  replace($('calibration-result'), calibrationResult(status));
}

function fillDefault(input, value) {
  if (input && !input.value && value) input.value = value;
}

function scheduleCalibrationPoll(running) {
  clearTimeout(calibrationTimer);
  if (running) calibrationTimer = setTimeout(refreshCalibration, CALIBRATION_POLL_MS);
}

function calibrationResult(status) {
  if (status.error) return h('p', { class: 'text-danger text-sm', text: status.error });
  if (status.running) {
    return h('div', { class: 'stack' }, [
      h('span', { class: 'text-sm', text: t('tools.calibration.running', { done: status.done, total: status.total }) }),
      progress(status.total ? status.done / status.total : 0, !status.total),
    ]);
  }
  const report = status.report;
  if (!report) return null;
  return h('div', { class: 'stack' }, [calibrationTable(report), calibrationBest(report)]);
}

function calibrationTable(report) {
  const columns = ['accept', 'reject', 'top1', 'core', 'fp', 'p95'];
  return h('div', { class: 'table-wrap' }, [
    h('table', { class: 'table' }, [
      h('thead', {}, [
        h(
          'tr',
          {},
          columns.map((column) => h('th', { scope: 'col', text: t(`tools.calibration.columns.${column}`) })),
        ),
      ]),
      h(
        'tbody',
        {},
        report.results.map((result) =>
          h('tr', {}, [
            h('td', { class: 'mono', text: result.accept_threshold.toFixed(2) }),
            h('td', { class: 'mono', text: result.reject_threshold.toFixed(2) }),
            h('td', { class: 'mono', text: fmtPct(result.top1_accuracy) }),
            h('td', { class: 'mono', text: fmtPct(result.core_accuracy) }),
            h('td', { class: 'mono', text: fmtPct(result.false_positive_rate) }),
            h('td', { class: 'mono', text: fmtMs(result.latency_p95_ms) }),
          ]),
        ),
      ),
    ]),
  ]);
}

function calibrationBest(report) {
  const best = report.best;
  if (!best) return empty(t('tools.calibration.noBest'));
  return h('div', { class: 'row' }, [
    h('span', {
      class: 'text-sm',
      text: t('tools.calibration.best', {
        accept: best.accept_threshold.toFixed(2),
        reject: best.reject_threshold.toFixed(2),
      }),
    }),
    h('button', {
      class: 'button button--primary button--sm',
      type: 'button',
      text: t('tools.calibration.apply'),
      onclick: () => applyThresholds(best),
    }),
  ]);
}

async function applyThresholds(best) {
  try {
    const response = await api('/api/config', {
      method: 'PUT',
      body: {
        patch: {
          intent: { accept_threshold: best.accept_threshold, reject_threshold: best.reject_threshold },
        },
      },
    });
    store.state.config = response.config;
    toast(t('tools.calibration.applied'), 'ok');
  } catch (error) {
    toast(t('tools.calibration.failed', { error: error.message }), 'error');
  }
}
