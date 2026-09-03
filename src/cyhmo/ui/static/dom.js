/* Construção de DOM e formatação — sem regra de negócio e sem estado. */

'use strict';

import { t } from './i18n.js';

export const $ = (id) => document.getElementById(id);

export function h(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
      else if (key === 'dataset') Object.assign(node.dataset, value);
      else if (value === true) node.setAttribute(key, '');
      else node.setAttribute(key, String(value));
    }
  }
  for (const child of [].concat(children || [])) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function replace(container, children) {
  if (container) container.replaceChildren(...[].concat(children || []).filter(Boolean));
}

export function show(node, visible) {
  if (node) node.hidden = !visible;
}

export function pill(text, modifier, extra) {
  return h('span', { class: `pill ${modifier || 'pill--muted'} ${extra || ''}`.trim(), text });
}

export function kvList(pairs, options) {
  const items = pairs
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) =>
      h('li', { class: 'list__item' }, [
        h('span', { class: 'list__key', text: key }),
        h('span', { class: `list__value ${options && options.mono ? 'mono' : ''}`.trim() }, [
          value instanceof Node ? value : String(value === null ? '—' : value),
        ]),
      ]),
    );
  return h('ul', { class: 'list' }, items);
}

export function metric(label, valueText, ratio, options) {
  const opts = options || {};
  const classes = ['metric'];
  if (opts.over) classes.push('metric--over');
  if (opts.variant) classes.push(`metric--${opts.variant}`);
  const width = Math.max(0, Math.min(100, ratio * 100));
  return h('div', { class: classes.join(' ') }, [
    h('div', { class: 'metric__header' }, [
      h('span', { class: 'metric__label', text: label, title: label }),
      h('span', { class: 'metric__value', text: valueText }),
    ]),
    h('div', { class: 'metric__bar' }, [h('div', { class: 'metric__bar-fill', style: `width: ${width}%` })]),
    opts.help ? h('div', { class: 'metric__help', text: opts.help }) : null,
  ]);
}

export function levelMeter(label) {
  return h(
    'div',
    { class: 'level-meter', role: 'meter', 'aria-label': label, 'aria-valuemin': 0, 'aria-valuemax': 1, 'aria-valuenow': 0 },
    [h('div', { class: 'level-meter__fill' }), h('div', { class: 'level-meter__peak' })],
  );
}

/* Escreve nos nós em posição: o nível chega dezenas de vezes por segundo e não pode
   agendar o render geral. Atinge todos os medidores presentes na página. */
export function updateLevelMeters(level) {
  const rms = Math.min(1, Number((level || {}).rms) || 0);
  const peak = Math.min(1, Number((level || {}).peak) || 0);
  document.querySelectorAll('.level-meter').forEach((meter) => {
    meter.querySelector('.level-meter__fill').style.width = `${rms * 100}%`;
    meter.querySelector('.level-meter__peak').style.left = `${peak * 100}%`;
    meter.setAttribute('aria-valuenow', String(rms));
  });
}

export function progress(ratio, indeterminate) {
  const width = Math.max(0, Math.min(100, (ratio || 0) * 100));
  return h('div', { class: `progress ${indeterminate ? 'progress--indeterminate' : ''}`.trim() }, [
    h('div', { class: 'progress__fill', style: indeterminate ? null : `width: ${width}%` }),
  ]);
}

export function empty(message) {
  return h('p', { class: 'empty', text: message });
}

export function preJson(value) {
  return h('pre', { class: 'pre', text: JSON.stringify(value, null, 2) });
}

export function fmtMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${Number(value).toFixed(0)} ms`;
}

export function fmtPct(value) {
  if (value === null || value === undefined) return '—';
  return `${(Number(value) * 100).toFixed(0)}%`;
}

export function fmtScore(value) {
  if (value === null || value === undefined) return '—';
  return Number(value).toFixed(3);
}

export function fmtClock(monotonicSeconds) {
  if (monotonicSeconds === null || monotonicSeconds === undefined) return '—';
  const total = Number(monotonicSeconds);
  const minutes = Math.floor(total / 60);
  const seconds = (total - minutes * 60).toFixed(1).padStart(4, '0');
  return `${minutes}:${seconds}`;
}

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];

export function fmtBytes(value) {
  let size = Number(value || 0);
  if (!size) return '';
  let unit = 0;
  while (size >= 1024 && unit < BYTE_UNITS.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 10 || unit === 0 ? 0 : 1)} ${BYTE_UNITS[unit]}`;
}

export function fmtHex(value) {
  if (value === null || value === undefined) return '—';
  return `0x${Number(value).toString(16).toUpperCase().padStart(8, '0')}`;
}

export function statusLabel(status) {
  return status ? t(`status.${status}`) : t('status.pending');
}

export const STATUS_PILL = {
  injected: 'pill--ok',
  no_command: 'pill--warning',
  blocked_cannot_talk: 'pill--warning',
  cancelled: 'pill--muted',
  error: 'pill--error',
  dry_run: 'pill--muted',
};

export const COMPONENT_PILL = {
  off: 'pill--muted',
  loading: 'pill--transcribing',
  ready: 'pill--ok',
  busy: 'pill--listening',
  error: 'pill--error',
};
