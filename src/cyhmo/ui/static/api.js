/* Cliente HTTP da interface: uma função, erros já legíveis.

   O servidor exige o token da sessão em toda rota /api. Ele chega num <meta> do HTML,
   então uma página de outra origem não consegue lê-lo — e o cabeçalho próprio ainda
   obriga o navegador ao preflight, que ela também não passa. */

'use strict';

import { t } from './i18n.js';

const TOKEN_META = 'meta[name="cyhmo-token"]';
const TOKEN_HEADER = 'X-CYHMO-Token';
const STALE_TOKEN = 'stale_token';
/* Reiniciar o mod sorteia um token novo e esta aba fica com o antigo. Recarregar busca o
   HTML atual e resolve; a janela impede que um servidor teimoso vire laço de recarga. */
const RELOAD_GUARD_KEY = 'cyhmo.tokenReload';
const RELOAD_GUARD_MS = 10000;

const token = (document.querySelector(TOKEN_META) || {}).content || '';

export function sessionToken() {
  return token;
}

export function renewSession() {
  let previous = 0;
  try {
    previous = Number(window.sessionStorage.getItem(RELOAD_GUARD_KEY)) || 0;
  } catch (_error) {
    previous = 0;
  }
  const now = Date.now();
  if (now - previous < RELOAD_GUARD_MS) return;
  try {
    window.sessionStorage.setItem(RELOAD_GUARD_KEY, String(now));
  } catch (_error) {
    /* aba anônima sem storage: recarregar uma vez ainda é melhor que travar */
  }
  window.location.reload();
}

export function handleStaleToken(payload) {
  if (!payload || payload.code !== STALE_TOKEN) return false;
  renewSession();
  return true;
}

export async function api(path, options) {
  const init = Object.assign({ headers: {} }, options || {});
  init.headers = Object.assign({ 'Content-Type': 'application/json', [TOKEN_HEADER]: token }, init.headers);
  if (init.body && typeof init.body !== 'string') init.body = JSON.stringify(init.body);
  let response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    throw new Error(t('common.noConnection', { error: error.message }));
  }
  let payload = null;
  try {
    payload = await response.json();
  } catch (_error) {
    payload = null;
  }
  if (!response.ok) {
    const expired = handleStaleToken(payload);
    const message = payload && (payload.error || (payload.detail && JSON.stringify(payload.detail)));
    const error = new Error(message || t('common.httpError', { status: response.status }));
    /* Marcado para quem estiver reconectando parar de tentar: a página já está sendo
       recarregada para buscar o token da sessão nova. */
    if (expired) error.sessionExpired = true;
    throw error;
  }
  return payload;
}
