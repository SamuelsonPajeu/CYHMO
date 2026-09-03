/* Histórico completo e detalhe de um enunciado (modo avançado). */

'use strict';

import {
  $,
  empty,
  fmtMs,
  fmtPct,
  fmtScore,
  h,
  kvList,
  metric,
  pill,
  preJson,
  replace,
  statusLabel,
  STATUS_PILL,
} from './dom.js';
import { t } from './i18n.js';
import { scheduleRender, store, totalMs } from './store.js';

const STAGE_KEYS = {
  vad_tail: 'vad',
  vad: 'vad',
  capture: 'capture',
  stt: 'stt',
  intent: 'intent',
  interpret: 'intent',
  llm: 'llm',
  state: 'state',
  inject: 'inject',
  state_inject: 'inject',
  reserve: 'reserve',
  total: 'total',
};

export function renderHistory(state) {
  const rows = state.history || [];
  replace($('history-rows'), rows.map((row) => historyRowNode(state, row)));
  const selected = rows.find((row) => row.utt_id === store.selectedUtterance);
  replace(
    $('history-detail'),
    selected
      ? [utteranceDetail(selected.detail), h('div', { class: 'stack' }, latencyMetrics(state, selected.detail))]
      : empty(t('history.empty')),
  );
}

function historyRowNode(state, row) {
  const selected = store.selectedUtterance === row.utt_id;
  const select = () => {
    store.selectedUtterance = row.utt_id;
    scheduleRender();
  };
  return h(
    'tr',
    {
      class: `table__row--clickable ${selected ? 'table__row--selected' : ''}`.trim(),
      tabindex: 0,
      onclick: select,
      onkeydown: (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          select();
        }
      },
    },
    [
      h('td', { class: 'mono', text: row.utt_id }),
      h('td', {}, [pill(statusLabel(row.status), STATUS_PILL[row.status])]),
      h('td', { class: 'td--wrap', text: row.text || t('event.empty') }),
      h('td', { class: 'td--wrap', text: (row.keys || []).join(', ') || '—' }),
      h('td', { text: t(`method.${row.method}`) }),
      h('td', { class: 'mono', text: fmtScore(row.confidence) }),
      h('td', {
        class: `mono ${row.over_budget ? 'text-warning' : ''}`.trim(),
        text: `${Number(row.total_ms).toFixed(0)}${row.over_budget ? ' ⚠' : ''}`,
      }),
    ],
  );
}

export function latencyMetrics(state, record) {
  const budget = state.budget_ms || {};
  const latency = record ? record.latency_ms || {} : {};
  const stages = [...Object.keys(budget), ...Object.keys(latency).filter((key) => !(key in budget))];
  const total = record ? totalMs(latency) : null;
  return stages.map((stage) => {
    const value = stage === 'total' && total !== null ? total : latency[stage];
    const limit = budget[stage];
    const ratio = limit ? Number(value || 0) / limit : 0;
    const over = limit !== undefined && value !== undefined && Number(value) > limit;
    return metric(stageLabel(stage), value === undefined ? '—' : fmtMs(value), ratio, {
      over,
      variant: stage === 'total' ? 'accent' : null,
      help: limit !== undefined ? t('tools.latency.budget', { limit }) : t('tools.latency.noBudget'),
    });
  });
}

function stageLabel(stage) {
  const key = STAGE_KEYS[stage];
  return key ? t(`stage.${key}`) : stage;
}

export function candidateMetrics(interpretation, commands) {
  return (interpretation.candidates || []).map((candidate) =>
    metric(candidate.key, fmtScore(candidate.score), Math.max(0, candidate.score), {
      variant: commands.some((command) => command.key === candidate.key) ? 'success' : null,
      help: candidate.matched_example
        ? t('history.candidateExample', { example: candidate.matched_example, lang: candidate.example_lang })
        : null,
    }),
  );
}

function utteranceDetail(record) {
  const interpretation = record.interpretation;
  const commands = interpretation ? interpretation.commands || [] : [];
  const injection = record.injection;
  const result = injection ? injection.result : null;
  const header = h('div', { class: 'row' }, [
    pill(statusLabel(record.status), record.status ? STATUS_PILL[record.status] : 'pill--listening'),
    h('span', { class: 'mono', text: record.utt_id }),
    h('span', {
      class: 'text-muted text-sm',
      text: t('history.audio', { source: record.source, duration: fmtMs(record.duration_ms) }),
    }),
    record.wav_path ? h('span', { class: 'mono text-muted', text: record.wav_path }) : null,
  ]);
  return h('div', { class: 'stack' }, [
    header,
    h('div', { class: 'grid' }, [
      heardBlock(record, interpretation),
      understoodBlock(interpretation, commands),
      injectedBlock(record, injection, result),
    ]),
  ]);
}

function heardBlock(record, interpretation) {
  const transcript = record.transcript;
  return h('div', { class: 'stack' }, [
    h('h3', { class: 'text-sm text-muted', text: t('history.heard') }),
    h('div', { class: 'quote', text: transcript && transcript.text ? transcript.text : t('event.empty') }),
    transcript
      ? kvList([
          [t('history.rawText'), transcript.raw_text || transcript.text],
          [t('history.normalized'), interpretation ? interpretation.normalized_text || '—' : '—'],
          [t('history.language'), transcript.lang || '—'],
          [t('history.sttConfidence'), fmtPct(transcript.confidence)],
        ])
      : h('p', { class: 'text-muted text-sm', text: t('history.noTranscript') }),
  ]);
}

function understoodBlock(interpretation, commands) {
  return h('div', { class: 'stack' }, [
    h('h3', { class: 'text-sm text-muted', text: t('history.understood') }),
    h(
      'div',
      { class: 'row' },
      commands.length
        ? commands.map((command) => pill(command.key, 'pill--listening', 'pill--lg'))
        : [pill(t('history.noCommand'), 'pill--muted', 'pill--lg')],
    ),
    interpretation
      ? kvList([
          [t('history.method'), t(`method.${interpretation.method}`)],
          [t('history.confidence'), fmtScore(interpretation.confidence)],
          [t('history.reason'), interpretation.reason || '—'],
          [t('history.latency'), fmtMs(interpretation.latency_ms)],
        ])
      : h('p', { class: 'text-muted text-sm', text: t('history.noInterpretation') }),
    interpretation && (interpretation.candidates || []).length
      ? h('div', { class: 'stack' }, [
          h('span', { class: 'text-sm text-muted', text: t('history.candidates') }),
          ...candidateMetrics(interpretation, commands),
        ])
      : null,
  ]);
}

function injectedBlock(record, injection, result) {
  return h('div', { class: 'stack' }, [
    h('h3', { class: 'text-sm text-muted', text: t('history.injected') }),
    injection
      ? h(
          'div',
          { class: 'row' },
          (injection.commands || []).map((command) =>
            pill(command.key, result && result.ok ? 'pill--injecting' : 'pill--error', 'pill--lg'),
          ),
        )
      : h('p', { class: 'text-muted text-sm', text: t('history.nothingInjected') }),
    result
      ? kvList([
          [t('history.write'), result.ok ? t('history.writeOk') : t('history.writeFailed', { error: result.error || '' })],
          [t('history.oracle'), oracleLabel(result.matched)],
          [t('history.latency'), fmtMs(result.latency_ms)],
        ])
      : null,
    result && result.payload_echo && Object.keys(result.payload_echo).length ? preJson(result.payload_echo) : null,
    record.error ? h('p', { class: 'text-danger text-sm', text: t('history.errorLine', { error: record.error }) }) : null,
  ]);
}

function oracleLabel(matched) {
  if (matched === null || matched === undefined) return t('history.oracleUnknown');
  return matched ? t('history.oracleMatched') : t('history.oracleMissed');
}
