/* Composição da View: carrega os textos, liga os módulos e mantém o WebSocket.
   Nenhuma regra de negócio — o snapshot do servidor sempre manda. */

'use strict';

import { api, sessionToken } from './api.js';
import { bindCheats, renderCheats } from './cheats.js';
import { $, COMPONENT_PILL, h, pill, replace, show, updateLevelMeters } from './dom.js';
import { renderHistory } from './history.js';
import { adopt, applyStaticText, t, translated } from './i18n.js';
import { bindConsole, renderConsole } from './logview.js';
import { renderPanel, updateVoiceLevel } from './panel.js';
import { renderSettings } from './settings.js';
import { applySnapshot, applyEvent, isAdvanced, isGrammarStale, refreshState, scheduleRender, setRenderer, store } from './store.js';
import { applyTheme } from './theme.js';
import { toast } from './toast.js';
import { bindTools, refreshCalibration, renderTools } from './tools.js';
import { adoptUpdate, openUpdate, renderUpdate, resetUpdate } from './update.js';

const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 10000;
const ADVANCED_TABS = new Set(['history', 'tools', 'console']);
/* Fechar o terminal mata o mod, mas a página continua desenhada e o usuário segue
   clicando nela. Depois desta espera sem socket, a página diz que não manda mais nada.
   O reinício ganha uma folga maior: ele passa pela carga dos modelos de novo. */
const OFFLINE_NOTICE_MS = 8000;
const RESTART_NOTICE_MS = 90000;

const LIFECYCLE = {
  restart: { confirm: 'app.restartConfirm', pending: 'app.restarting', failed: 'app.restartFailed' },
  exit: { confirm: 'app.exitConfirm', pending: 'app.exited', failed: 'app.exitFailed' },
};
const RECONNECTING = new Set(['restart', 'update']);

let reconnectDelay = RECONNECT_MIN_MS;
let offlineTimer = null;

function render() {
  const state = store.state;
  if (!state) return;
  applyTheme((state.config.ui || {}).theme);
  renderHeader(state);
  renderTabs();
  renderUpdate(state);
  renderOffline();
  if (store.activeTab === 'panel') renderPanel(state);
  if (store.activeTab === 'cheats') renderCheats(state);
  if (store.activeTab === 'history') renderHistory(state);
  if (store.activeTab === 'tools') renderTools(state);
  if (store.activeTab === 'console') renderConsole(state);
  if (store.activeTab === 'settings' && store.renderedSettingsVersion !== store.settingsVersion) {
    renderSettings(state);
    store.renderedSettingsVersion = store.settingsVersion;
  }
}

function renderHeader(state) {
  renderLink(state);
  const advanced = isAdvanced();
  const phase = $('phase-pill');
  const current = state.phase || 'idle';
  show(phase, advanced);
  phase.className = `pill pill--${current}`;
  phase.textContent = t(`phase.${current}`);
  show($('component-status'), advanced);
  if (advanced) renderComponents(state);
  const listen = $('listen-toggle');
  listen.textContent = t(state.listening ? 'listen.stop' : 'listen.start');
  listen.className = `button ${state.listening ? 'button--secondary' : 'button--primary'}`;
  show($('restart-badge'), Boolean(state.restart_pending));
}

function renderLink(state) {
  const node = $('link-pill');
  const [text, modifier, title] = linkState(state);
  node.className = `pill pill--lg ${modifier}`;
  node.textContent = text;
  node.title = title;
}

function linkState(state) {
  if (store.lifecycle === 'exit') return [t('connection.closed'), 'pill--muted', ''];
  if (store.socketDown) return [t('connection.offline'), 'pill--error', ''];
  const pine = (state.components || {}).pine || {};
  if (pine.status === 'loading' || !pine.status) return [t('connection.checking'), 'pill--muted', ''];
  if (pine.status !== 'ready') return [t('connection.unlinked'), 'pill--warning', t('connection.unlinkedTitle')];
  if (!(state.grammar || {}).version) return [t('connection.noScene'), 'pill--muted', t('connection.noSceneTitle')];
  if (isGrammarStale(state)) return [t('connection.stale'), 'pill--warning', t('connection.staleTitle')];
  return [t('connection.linked'), 'pill--ok', t('connection.linkedTitle')];
}

function renderComponents(state) {
  replace(
    $('component-status'),
    Object.entries(state.components || {}).map(([name, info]) => {
      const node = pill(`${t(`component.${name}`)}: ${info.status}`, COMPONENT_PILL[info.status]);
      if (info.detail) node.title = info.detail;
      return node;
    }),
  );
}

/* Encerrar pelo botão Sair é escolha do usuário e a página diz isso na hora; perder o
   socket sem aviso é o terminal fechado, e aí a página precisa parar de fingir que
   controla alguma coisa. */
function renderOffline() {
  const kind = store.lifecycle === 'exit' ? 'closed' : store.stopped ? 'lost' : '';
  replace($('offline-root'), kind ? [offlineDialog(kind)] : []);
}

function offlineDialog(kind) {
  const lost = kind === 'lost';
  return h('div', { class: 'modal', role: 'alertdialog', 'aria-modal': 'true' }, [
    h('div', { class: 'modal__dialog' }, [
      h('h2', { class: 'modal__title', text: t(`offline.${kind}Title`) }),
      h('p', { text: t(`offline.${kind}`) }),
      lost ? h('p', { class: 'text-muted text-sm', text: t('offline.hint') }) : null,
      lost
        ? h('div', { class: 'modal__actions' }, [
            h('button', {
              class: 'button button--primary',
              type: 'button',
              text: t('offline.retry'),
              onclick: () => window.location.reload(),
            }),
          ])
        : null,
    ]),
  ]);
}

function armOfflineNotice() {
  if (offlineTimer !== null) return;
  offlineTimer = setTimeout(
    () => {
      offlineTimer = null;
      store.stopped = true;
      scheduleRender();
    },
    RECONNECTING.has(store.lifecycle) ? RESTART_NOTICE_MS : OFFLINE_NOTICE_MS,
  );
}

function clearOfflineNotice() {
  clearTimeout(offlineTimer);
  offlineTimer = null;
  store.stopped = false;
}

function renderTabs() {
  const advanced = isAdvanced();
  document.querySelectorAll('.tabs__tab').forEach((button) => {
    show(button, advanced || !ADVANCED_TABS.has(button.dataset.tab));
  });
  if (!advanced && ADVANCED_TABS.has(store.activeTab)) activateTab('panel');
}

function activateTab(name) {
  store.activeTab = name;
  document.querySelectorAll('.tabs__tab').forEach((button) => {
    const active = button.dataset.tab === name;
    button.classList.toggle('tabs__tab--active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  document.querySelectorAll('.panel').forEach((panel) => {
    panel.hidden = panel.id !== `tab-${name}`;
  });
  if (name === 'tools') refreshCalibration();
  scheduleRender();
}

function bindHeader() {
  $('listen-toggle').addEventListener('click', toggleListening);
  $('update-badge').addEventListener('click', openUpdate);
  $('restart-badge').addEventListener('click', () => requestLifecycle('restart'));
  $('app-restart').addEventListener('click', () => requestLifecycle('restart'));
  $('app-exit').addEventListener('click', () => requestLifecycle('exit'));
  document.querySelectorAll('.tabs__tab').forEach((button) => {
    button.addEventListener('click', () => activateTab(button.dataset.tab));
  });
}

async function toggleListening() {
  const button = $('listen-toggle');
  const listening = store.state ? store.state.listening : false;
  button.disabled = true;
  try {
    const response = await api(listening ? '/api/listen/stop' : '/api/listen/start', { method: 'POST' });
    if (store.state) store.state.listening = response.listening;
    toast(t(response.listening ? 'listen.started' : 'listen.stopped'), 'ok');
    scheduleRender();
  } catch (error) {
    toast(t('listen.failed', { error: error.message }), 'error');
  } finally {
    button.disabled = false;
  }
}

async function requestLifecycle(action) {
  const spec = LIFECYCLE[action];
  if (!window.confirm(t(spec.confirm))) return;
  const buttons = [$('app-restart'), $('app-exit')];
  buttons.forEach((button) => {
    button.disabled = true;
  });
  try {
    await api(`/api/${action}`, { method: 'POST' });
    store.lifecycle = action;
    toast(t(spec.pending), 'warning');
  } catch (error) {
    toast(t(spec.failed, { error: error.message }), 'error');
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

/* O token vai na query porque o navegador não deixa a página escolher cabeçalhos do
   WebSocket; do lado do servidor ele é conferido igual ao das rotas /api. */
function connect() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${protocol}://${location.host}/ws?token=${encodeURIComponent(sessionToken())}`);
  store.socket = socket;
  socket.addEventListener('open', onSocketOpen);
  socket.addEventListener('message', onSocketMessage);
  socket.addEventListener('close', onSocketClose);
  socket.addEventListener('error', () => socket.close());
}

function onSocketOpen() {
  reconnectDelay = RECONNECT_MIN_MS;
  const wasDown = store.socketDown;
  const lifecycle = store.lifecycle;
  store.socketDown = false;
  clearOfflineNotice();
  if (RECONNECTING.has(lifecycle)) {
    store.lifecycle = null;
    resetUpdate();
    [$('app-restart'), $('app-exit')].forEach((button) => {
      button.disabled = false;
    });
    toast(t(lifecycle === 'update' ? 'update.done' : 'app.restarted'), 'ok');
  } else if (wasDown) {
    toast(t('connection.back'), 'ok');
  }
  scheduleRender();
}

function onSocketMessage(message) {
  let payload;
  try {
    payload = JSON.parse(message.data);
  } catch (_error) {
    return;
  }
  if (payload.kind === 'snapshot') {
    applySnapshot(payload.state);
    adoptUpdate();
    return;
  }
  if (applyEvent(payload) === 'level') applyLevel();
  else scheduleRender();
}

function onSocketClose() {
  if (store.lifecycle === 'exit') {
    scheduleRender();
    return;
  }
  store.socketDown = true;
  armOfflineNotice();
  scheduleRender();
  const delay = RECONNECTING.has(store.lifecycle) ? RECONNECT_MIN_MS : reconnectDelay;
  reconnectDelay = Math.min(RECONNECT_MAX_MS, delay * 2);
  setTimeout(reconnect, delay);
}

/* O navegador não distingue servidor fora do ar de token recusado: o handshake negado
   chega como fechamento anormal nos dois casos, sem código de motivo. Ler o `code` do
   evento era o que deixava a aba reconectando para sempre depois de um reinício, porque
   o 1008 do servidor nunca chegava até aqui.

   Quem sabe a diferença é a API, então toda tentativa começa por ela: com o mod ainda
   subindo dá erro de rede e o socket tenta assim mesmo; com token de uma sessão que já
   morreu, `api()` recarrega a página e não há por que abrir socket nenhum. */
async function reconnect() {
  try {
    await api('/api/state');
  } catch (error) {
    if (error && error.sessionExpired) return;
  }
  connect();
}

/* Nível do microfone não passa pelo render geral. */
function applyLevel() {
  updateVoiceLevel(store.state);
  updateLevelMeters(store.state.level);
}

async function boot() {
  adopt(await api('/api/i18n'));
  applyStaticText(document);
  setRenderer(render);
  bindHeader();
  bindCheats();
  bindConsole();
  bindTools();
  try {
    await refreshState();
  } catch (error) {
    toast(t('common.stateFailed', { error: error.message }), 'error');
  }
  connect();
}

/* Falhar aqui deixava a página em branco, sem uma palavra sobre o motivo — o pior
   estado possível. Os textos de reserva são literais porque este é justamente o
   caminho em que o locale pode não ter chegado. */
function showBootFailure(error) {
  const shell = document.querySelector('.app-shell__main') || document.body;
  replace(shell, [
    h('section', { class: 'card' }, [
      h('h2', { class: 'card__title', text: translated('app.bootFailed', 'Interface failed to start') }),
      h('div', { class: 'card__body' }, [
        h('p', { class: 'text-danger', text: String((error && error.message) || error) }),
        h('p', {
          class: 'text-muted text-sm',
          text: translated('app.bootFailedHint', 'Reload the page (Ctrl+Shift+R). If it persists, restart the mod.'),
        }),
      ]),
    ]),
  ]);
}

boot().catch(showBootFailure);
