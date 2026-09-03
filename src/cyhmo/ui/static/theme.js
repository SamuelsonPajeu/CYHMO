/* Tema: só troca o atributo que redefine os tokens. */

'use strict';

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme || 'auto';
}
