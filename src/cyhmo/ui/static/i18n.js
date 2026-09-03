/* Textos da interface: tudo sai de /api/i18n, nada é escrito na View. */

'use strict';

const PLACEHOLDER = /\{(\w+)\}/g;

let strings = {};
let language = 'en';
let catalog = [];

function flatten(tree, prefix, into) {
  for (const [key, value] of Object.entries(tree)) {
    const dotted = `${prefix}${key}`;
    if (value && typeof value === 'object') flatten(value, `${dotted}.`, into);
    else into[dotted] = String(value);
  }
  return into;
}

export function t(key, params) {
  const template = strings[key];
  if (template === undefined) return key;
  if (!params) return template;
  return template.replace(PLACEHOLDER, (match, name) => (name in params ? String(params[name]) : match));
}

/* Para o punhado de textos que precisam existir mesmo quando o locale não chegou. */
export function translated(key, fallback) {
  return key in strings ? strings[key] : fallback;
}

export function availableLanguages() {
  return catalog;
}

export function adopt(bundle) {
  language = bundle.language || 'en';
  catalog = bundle.available || [];
  strings = flatten(bundle.strings || {}, '', {});
  document.documentElement.lang = t('meta.html_lang') || language;
}

const TRANSLATED_ATTRIBUTES = [
  ['data-i18n-placeholder', 'i18nPlaceholder', 'placeholder'],
  ['data-i18n-title', 'i18nTitle', 'title'],
  ['data-i18n-label', 'i18nLabel', 'aria-label'],
];

/* Marcadores no HTML em vez de texto: a estrutura fica legível e nada de pt-BR
   sobrevive escondido num atributo. */
export function applyStaticText(root) {
  const scope = root || document;
  scope.querySelectorAll('[data-i18n]').forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  for (const [selector, dataset, attribute] of TRANSLATED_ATTRIBUTES) {
    scope.querySelectorAll(`[${selector}]`).forEach((node) => {
      node.setAttribute(attribute, t(node.dataset[dataset]));
    });
  }
}
