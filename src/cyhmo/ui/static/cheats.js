/* Cheats: os comandos que a cena aceita agora, um clique para enviar sem falar. */

'use strict';

import { api } from './api.js';
import { $, empty, h, replace, show } from './dom.js';
import { t } from './i18n.js';
import { isGrammarStale, store } from './store.js';
import { toast } from './toast.js';

const SEARCH_DEBOUNCE_MS = 180;
const LIST_LIMIT = 5000;

let searchTimer = null;
let lastQuery = '';
let lastVersion = -1;

export function renderCheats(state) {
  const grammar = state.grammar || {};
  show($('cheats-stale'), isGrammarStale(state) && Number(grammar.size) > 0);
  if (grammar.version === lastVersion) return;
  lastVersion = grammar.version;
  loadEntries(lastQuery);
}

export function bindCheats() {
  $('cheats-search').addEventListener('input', (event) => {
    lastQuery = event.target.value;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadEntries(lastQuery), SEARCH_DEBOUNCE_MS);
  });
}

async function loadEntries(query) {
  const list = $('cheats-list');
  try {
    const payload = await api(`/api/grammar?q=${encodeURIComponent(query)}&limit=${LIST_LIMIT}`);
    const total = (store.state && store.state.grammar && store.state.grammar.size) || payload.count;
    $('cheats-count').textContent = query
      ? t('cheats.countFiltered', { count: payload.count, total })
      : t('cheats.count', { count: payload.count });
    if (!payload.count) {
      replace(list, empty(t(query ? 'cheats.empty' : 'cheats.noGrammar')));
      return;
    }
    replace(list, payload.entries.map(entryNode));
  } catch (error) {
    replace(list, empty(error.message));
  }
}

function entryNode(entry) {
  return h(
    'li',
    {
      class: 'list__item list__item--action',
      tabindex: 0,
      onclick: () => send(entry),
      onkeydown: (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          send(entry);
        }
      },
    },
    [entry],
  );
}

async function send(command) {
  try {
    const result = await api('/api/inject', { method: 'POST', body: { text: command } });
    if (!result.ok) toast(t('cheats.failed', { command, error: result.error || '' }), 'error');
    else if (result.matched === false) toast(t('cheats.ignored', { command }), 'warning');
    else toast(t('cheats.sent', { command }), 'ok');
  } catch (error) {
    toast(t('cheats.failed', { command, error: error.message }), 'error');
  }
}
