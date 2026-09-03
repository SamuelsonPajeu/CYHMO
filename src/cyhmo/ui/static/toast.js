/* Avisos efêmeros no canto da tela. */

'use strict';

import { $, h } from './dom.js';

const LIFETIME_MS = 4500;

export function toast(message, kind) {
  const node = h('div', { class: `toast toast--${kind || 'info'}`, role: 'status', text: message });
  $('toast-stack').appendChild(node);
  setTimeout(() => node.remove(), LIFETIME_MS);
}
