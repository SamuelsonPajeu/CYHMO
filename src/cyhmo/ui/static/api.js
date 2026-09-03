/* Cliente HTTP da interface: uma função, erros já legíveis. */

'use strict';

import { t } from './i18n.js';

export async function api(path, options) {
  const init = Object.assign({ headers: { 'Content-Type': 'application/json' } }, options || {});
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
    const message = payload && (payload.error || (payload.detail && JSON.stringify(payload.detail)));
    throw new Error(message || t('common.httpError', { status: response.status }));
  }
  return payload;
}
