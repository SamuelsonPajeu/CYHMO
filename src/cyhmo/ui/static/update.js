/* Aviso de versão nova: o mod baixa, troca os arquivos e volta sozinho.
   O servidor é a fonte da verdade; aqui só ficam a escolha do usuário e o polling. */

'use strict';

import { api } from './api.js';
import { $, h, progress, replace, show } from './dom.js';
import { t } from './i18n.js';
import { scheduleRender, store } from './store.js';
import { toast } from './toast.js';

const POLL_MS = 1000;
const TRANSIENT = new Set(['checking', 'downloading', 'staging']);
const INSTALLING = new Set(['downloading', 'staging', 'ready']);

let live = null;
let reopened = false;
let suspended = false;
let timer = null;

export function renderUpdate(state) {
  const update = updateInfo(state);
  const open = visible(update);
  /* A instalação pode falhar dentro da thread do servidor: sem soltar o ciclo de vida
     aqui, a próxima reconexão anunciaria uma atualização que não aconteceu. */
  if (store.lifecycle === 'update' && update.phase === 'error') store.lifecycle = null;
  show($('update-badge'), notified(update) && !open);
  replace($('update-root'), open ? [dialog(update)] : []);
  schedulePoll(update);
}

/* "Pular esta versão" some com o aviso inteiro; "lembrar mais tarde" fecha só o diálogo
   e deixa o selo no cabeçalho. */
function notified(update) {
  return Boolean(update.available) && update.latest !== update.skipped;
}

/* O snapshot chega depois de toda reconexão e é o único que sabe a versão que está
   rodando agora — depois de uma atualização, o estado local aponta para a versão velha.
   Até ele chegar nada é mostrado, senão o aviso da versão recém-instalada pisca de volta. */
export function resetUpdate() {
  live = null;
  reopened = false;
  suspended = true;
}

export function adoptUpdate() {
  suspended = false;
}

export function updateInfo(state) {
  return live || (state && state.update) || {};
}

export function openUpdate() {
  reopened = true;
  refreshView();
}

export async function checkUpdate() {
  try {
    live = await api('/api/update/check', { method: 'POST' });
  } catch (error) {
    toast(t('update.checkFailed', { error: error.message }), 'error');
  }
  refreshView();
}

function visible(update) {
  if (suspended) return false;
  if (INSTALLING.has(update.phase)) return true;
  return Boolean(update.prompt) || (reopened && Boolean(update.available));
}

function dialog(update) {
  return h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'update-title' }, [
    h('div', { class: 'modal__dialog' }, [
      h('h2', { class: 'modal__title', id: 'update-title', text: t('update.title') }),
      h('p', { text: t('update.lead', { current: update.current, latest: update.latest }) }),
      update.notes ? h('pre', { class: 'pre', text: update.notes }) : null,
      INSTALLING.has(update.phase) ? working(update) : choices(update),
    ]),
  ]);
}

/* Sem botão de fechar durante a troca: o mod já está reiniciando sozinho e cancelar no
   meio deixaria meia instalação preparada. */
function working(update) {
  const percent = Math.round(Number(update.percent) || 0);
  return h('div', { class: 'stack' }, [
    h('p', { class: 'text-muted text-sm', text: t(`update.phase.${update.phase}`, { percent }) }),
    progress(percent / 100, update.phase !== 'downloading'),
  ]);
}

function choices(update) {
  return h('div', { class: 'stack' }, [
    update.error ? h('p', { class: 'field__error', text: update.error }) : null,
    h('div', { class: 'modal__actions' }, [
      button('update.later', 'button--ghost', postpone),
      button('update.skip', 'button--secondary', skip),
      button('update.install', 'button--primary', install),
    ]),
    releaseLink(update),
  ]);
}

function button(key, variant, onClick) {
  return h('button', { class: `button ${variant}`, type: 'button', text: t(key), onclick: onClick });
}

function releaseLink(update) {
  if (!update.url) return null;
  return h('a', {
    class: 'text-muted text-sm',
    href: update.url,
    target: '_blank',
    rel: 'noreferrer',
    text: t('update.release'),
  });
}

async function install() {
  store.lifecycle = 'update';
  try {
    live = await api('/api/update/install', { method: 'POST' });
  } catch (error) {
    store.lifecycle = null;
    toast(t('update.installFailed', { error: error.message }), 'error');
  }
  refreshView();
}

async function postpone() {
  reopened = false;
  await dismiss('/api/update/postpone');
}

async function skip() {
  reopened = false;
  await dismiss('/api/update/skip');
}

async function dismiss(path) {
  try {
    live = await api(path, { method: 'POST' });
  } catch (error) {
    toast(t('update.dismissFailed', { error: error.message }), 'error');
  }
  refreshView();
}

/* Um timer por vez: o render geral roda a cada evento do pipeline e reagendar em todos
   eles adiaria a consulta para sempre. */
function schedulePoll(update) {
  if (!TRANSIENT.has(update.phase)) {
    clearTimeout(timer);
    timer = null;
    return;
  }
  if (timer === null) timer = setTimeout(poll, POLL_MS);
}

async function poll() {
  timer = null;
  try {
    live = await api('/api/update');
  } catch (_error) {
    return;
  }
  refreshView();
}

/* O cartão de Configurações também mostra a versão, e ele só é redesenhado quando a
   versão do formulário muda. */
function refreshView() {
  if (!store.hotkeyRecording) store.settingsVersion += 1;
  scheduleRender();
}
