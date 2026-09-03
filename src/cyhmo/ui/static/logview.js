/* Console de eventos: anexa a linha nova, poda a antiga, não mexe no meio —
   reconstruir apagaria a seleção do usuário antes de ele soltar o mouse. */

'use strict';

import { $, fmtClock, h, replace } from './dom.js';
import { t } from './i18n.js';
import { scheduleRender, store } from './store.js';
import { toast } from './toast.js';

const LEVEL_ORDER = { debug: 0, info: 1, warning: 2, error: 3 };
const PIN_TOLERANCE_PX = 8;

export function bindConsole() {
  $('console-filter').addEventListener('change', (event) => {
    store.consoleLevel = event.target.value;
    scheduleRender();
  });
  $('console-copy').addEventListener('click', () => copyConsole(store.state || {}));
}

export function renderConsole(state) {
  const lines = visibleLines(state);
  const container = $('console-lines');
  const last = lines.length ? lines[lines.length - 1].seq : 0;
  const rendered = store.renderedConsole;
  if (rendered && rendered.filter === store.consoleLevel && rendered.last === last) return;
  $('console-count').textContent = t('console.count', { shown: lines.length, total: (state.console || []).length });
  const pinned = container.scrollTop + container.clientHeight >= container.scrollHeight - PIN_TOLERANCE_PX;
  if (!rendered || rendered.filter !== store.consoleLevel) {
    replace(container, lines.map(lineNode));
  } else {
    const first = lines.length ? lines[0].seq : Infinity;
    while (container.firstChild && Number(container.firstChild.dataset.seq) < first) container.firstChild.remove();
    container.append(...lines.filter((line) => line.seq > rendered.last).map(lineNode));
  }
  store.renderedConsole = { filter: store.consoleLevel, last };
  if (pinned) container.scrollTop = container.scrollHeight;
}

function visibleLines(state) {
  const minimum = LEVEL_ORDER[store.consoleLevel] || 0;
  return (state.console || []).filter((line) => (LEVEL_ORDER[line.level] || 0) >= minimum);
}

function lineNode(line) {
  return h('div', { class: `console__line console__line--${line.level || 'info'}`, dataset: { seq: line.seq } }, [
    h('span', { class: 'console__time', text: fmtClock(line.t) }),
    h('span', { text: line.level }),
    h('span', { class: 'console__source', text: line.source || '' }),
    h('span', { class: 'console__message', text: line.message }),
  ]);
}

async function copyConsole(state) {
  const text = visibleLines(state)
    .map((line) => `${fmtClock(line.t)} ${line.level} ${line.source || ''} ${line.message}`)
    .join('\n');
  try {
    await navigator.clipboard.writeText(text);
    toast(t('console.copied'), 'ok');
  } catch (_error) {
    toast(t('console.copyFailed'), 'error');
  }
}
