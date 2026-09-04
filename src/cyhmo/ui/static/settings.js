/* Configurações: o básico é o que o jogador precisa e salva sozinho;
   o avançado espelha o config.toml e salva por seção. */

'use strict';

import { api } from './api.js';
import { $, empty, fmtBytes, h, levelMeter, pill, progress, replace, show } from './dom.js';
import { availableLanguages, t } from './i18n.js';
import { getPath, isAdvanced, scheduleRender, setPath, store } from './store.js';
import { applyTheme } from './theme.js';
import { toast } from './toast.js';
import { checkUpdate, openUpdate, updateInfo } from './update.js';

const DEVICE_PATH = 'audio.device';
const SYSTEM_DEFAULT_DEVICE = 'default';
const HOTKEY_TIMEOUT_S = 10;
const LLM_POLL_MS = 1200;
const OTHER_MODEL = '__other__';
const CLIPPING = 0.99;
const TOO_QUIET = 0.01;

const RESTART_SECTIONS = new Set(['audio', 'activation', 'stt', 'languages', 'pine', 'state']);
const RESTART_INTENT_FIELDS = new Set(['embedding_backend', 'embedding_model', 'embedding_cache', 'annex']);

const ADVANCED_SECTIONS = [
  { id: 'audio', title: 'settings.advanced.audio', paths: ['audio'] },
  { id: 'activation', title: 'settings.advanced.activation', paths: ['activation', 'activation.vad'] },
  { id: 'stt', title: 'settings.advanced.stt', paths: ['stt', 'stt.whisper_cpp'] },
  { id: 'languages', title: 'settings.advanced.languages', paths: ['languages'], languages: true },
  { id: 'intent', title: 'settings.advanced.intent', paths: ['intent'] },
  { id: 'llm', title: 'settings.advanced.llm', paths: ['intent.llm'] },
  { id: 'pine', title: 'settings.advanced.pine', paths: ['pine', 'inject'] },
  { id: 'ui', title: 'settings.advanced.ui', paths: ['ui', 'update', 'debug'] },
];

let llmStatus = null;
let llmTimer = null;
let whisperStatus = null;
let whisperTimer = null;
let gpuWanted = false;
let micMeterVisible = false;
let registry = null;
let registryQuery = '';
let pullChoice = '';
let pullCustom = '';

export function renderSettings(state) {
  const advanced = isAdvanced();
  replace($('settings-root'), advanced ? ADVANCED_SECTIONS.map((section) => advancedCard(state, section)) : basicCards(state));
  show($('settings-advanced-note'), advanced);
  if (!(state.devices || []).length) refreshDevices();
  if (!advanced) {
    refreshLlm();
    refreshWhisper();
  }
  renderRegistry();
}

/* ---------- básico ---------- */

/* Onde o reconhecimento roda mora junto do peso, no cartão do modelo: é a mesma pergunta
   vista de dois lados — o peso grande só se paga com a placa de vídeo fazendo a conta. */
function basicCards(state) {
  return [
    micCard(state),
    activationCard(state),
    languageCard(state),
    sttModelCard(),
    appearanceCard(state),
    assistantCard(),
    updatesCard(state),
  ];
}

function card(titleKey, children) {
  return h('section', { class: 'card grid__span-2' }, [
    h('h2', { class: 'card__title', text: t(titleKey) }),
    h('div', { class: 'card__body' }, children),
  ]);
}

function settingRow(labelText, controls, note) {
  return h('div', { class: 'setting' }, [
    h('span', { class: 'setting__label', text: labelText }),
    h('div', { class: 'setting__control' }, controls),
    note ? h('span', { class: 'setting__note', text: note }) : null,
  ]);
}

function selectControl(value, options, onChange) {
  const node = h(
    'select',
    { class: 'field__control', onchange: (event) => onChange(event.target.value) },
    options.map((option) => h('option', { value: option.value, text: option.label, disabled: Boolean(option.disabled) })),
  );
  const current = value === null || value === undefined ? '' : String(value);
  if (current && !options.some((option) => option.value === current)) {
    node.appendChild(h('option', { value: current, text: current }));
  }
  node.value = current;
  return node;
}

function toggleControl(checked, onChange) {
  const input = h('input', { class: 'toggle__input', type: 'checkbox', onchange: (event) => onChange(event.target.checked) });
  input.checked = Boolean(checked);
  return h('label', { class: 'toggle' }, [input]);
}

function micCard(state) {
  const current = getPath(state.config, DEVICE_PATH);
  const options = [{ value: SYSTEM_DEFAULT_DEVICE, label: t('settings.mic.deviceDefault') }];
  (state.devices || []).forEach((device) => options.push({ value: device.name, label: device.name }));
  if (current && !options.some((option) => option.value === String(current))) {
    options.push({ value: String(current), label: t('settings.mic.deviceMissing', { name: current }) });
  }
  const result = h('span', { class: 'setting__note' });
  const meter = levelMeter(t('settings.mic.level'));
  show(meter, micMeterVisible);
  const test = h('button', {
    class: 'button button--secondary',
    type: 'button',
    text: t('settings.mic.test'),
    onclick: () => runMicTest(test, result, meter),
  });
  return card('settings.mic.title', [
    settingRow(t('settings.mic.device'), [
      selectControl(current, options, (value) => save({ audio: { device: value } }, 'settings.mic.title')),
      test,
    ]),
    meter,
    result,
  ]);
}

/* O medidor some ao fim do teste: ele é alimentado pelo nível do microfone que chega
   o tempo todo, então deixá-lo na tela sugere que o mod continua escutando ali. */
async function runMicTest(button, output, meter) {
  button.disabled = true;
  micMeterVisible = true;
  show(meter, true);
  output.textContent = t('settings.mic.testing');
  try {
    const result = await api('/api/mic-test', { method: 'POST', body: { seconds: 2 } });
    if (result.peak > CLIPPING) output.textContent = t('settings.mic.testClipping');
    else if (result.rms < TOO_QUIET) output.textContent = t('settings.mic.testLow');
    else output.textContent = t('settings.mic.testOk');
  } catch (error) {
    output.textContent = t('settings.mic.testFailed', { error: error.message });
  } finally {
    button.disabled = false;
    micMeterVisible = false;
    show(meter, false);
  }
}

/* Só push-to-talk no básico: o modo mãos-livres continua existindo no avançado, mas
   escolher entre os dois é decisão de quem foi atrás dela. */
function activationCard(state) {
  return card('settings.activation.title', [hotkeyRow((state.config.activation || {}).ptt_hotkey)]);
}

function hotkeyRow(current) {
  const label = h('span', { class: 'mono', text: current || '—' });
  const button = h('button', {
    class: 'button button--secondary',
    type: 'button',
    text: t('settings.activation.record'),
  });
  button.addEventListener('click', () => captureHotkey(button, label));
  return settingRow(t('settings.activation.hotkey'), [label, button]);
}

/* aria-disabled em vez de disabled: o botão precisa manter o foco durante a gravação.
   A trava também segura o re-render das Configurações, que descartaria a tecla capturada. */
async function captureHotkey(button, label) {
  if (store.hotkeyRecording) return;
  store.hotkeyRecording = true;
  button.setAttribute('aria-disabled', 'true');
  button.classList.add('button--recording');
  button.textContent = t('settings.activation.recording');
  button.focus();
  try {
    const response = await api('/api/hotkey/capture', { method: 'POST', body: { timeout_s: HOTKEY_TIMEOUT_S } });
    if (!response.accepted) {
      toast(t('settings.activation.recordNone'), 'warning');
      return;
    }
    label.textContent = response.key;
    await save({ activation: { ptt_hotkey: response.key } }, 'settings.activation.title');
    toast(t('settings.activation.recorded', { key: response.key }), 'ok');
  } catch (error) {
    toast(t('settings.activation.recordFailed', { error: error.message }), 'error');
  } finally {
    store.hotkeyRecording = false;
    button.removeAttribute('aria-disabled');
    button.classList.remove('button--recording');
    button.textContent = t('settings.activation.record');
  }
}

function languageCard(state) {
  const uiOptions = [{ value: 'auto', label: t('settings.language.interfaceAuto') }];
  availableLanguages().forEach((entry) => uiOptions.push({ value: entry.code, label: entry.name }));
  return card('settings.language.title', [
    h('div', { class: 'stack' }, [
      h('span', { class: 'setting__label', text: t('settings.language.spoken') }),
      languageOrderControl(state, {
        onChange: (order) => save({ languages: { enabled: order, primary: order[0] } }, 'settings.language.title'),
      }),
      h('span', { class: 'setting__note', text: t('settings.language.orderHelp') }),
    ]),
    settingRow(t('settings.language.interface'), [
      selectControl((state.config.ui || {}).language, uiOptions, async (value) => {
        if (await save({ ui: { language: value } }, 'settings.language.title')) window.location.reload();
      }),
    ]),
  ]);
}

/* Baixar idioma novo mora só no avançado: no básico a lista de idiomas instalados
   basta, e a busca ocupava espaço oferecendo o que quase nunca é preciso. */
function registrySearch() {
  return h('div', { class: 'stack' }, [
    h('span', { class: 'setting__label', text: t('settings.language.add') }),
    h('input', {
      class: 'field__control',
      type: 'search',
      placeholder: t('settings.language.addPlaceholder'),
      autocomplete: 'off',
      oninput: (event) => {
        registryQuery = event.target.value;
        renderRegistry();
      },
    }),
    h('div', { id: 'registry-body', class: 'stack' }),
  ]);
}

/* ---------- lista ordenada de idiomas ---------- */

/* Widget único do básico e do avançado: a ordem É a configuração — o primeiro da
   lista vira o primário do reconhecedor e os demais entram no matching nessa ordem.
   A ordem corrente fica no próprio nó (`data-lang-order`) porque o formulário do
   avançado só a lê ao gravar, enquanto o básico grava a cada mexida. */
function languageOrderControl(state, options) {
  const opts = options || {};
  const root = h('div', { class: 'stack', 'data-lang-order': '' });
  let order = enabledOrder(state);

  const draw = () => {
    root.dataset.langOrder = order.join(' ');
    replace(root, [h('ul', { class: 'list' }, order.map(orderItem)), addRow()]);
  };
  const apply = (next) => {
    order = next;
    draw();
    if (opts.onChange) opts.onChange(next);
  };

  function orderItem(code, index) {
    const move = (delta) => {
      const next = [...order];
      next.splice(index + delta, 0, next.splice(index, 1)[0]);
      apply(next);
    };
    return h('li', { class: 'list__item list__item--model' }, [
      h('span', { text: packLabel(state, code, opts.showCodes) }),
      index === 0 ? pill(t('settings.language.primaryBadge'), 'pill--ok') : null,
      orderButton('↑', 'settings.language.moveUp', index === 0, () => move(-1)),
      orderButton('↓', 'settings.language.moveDown', index === order.length - 1, () => move(1)),
      orderButton('✕', order.length === 1 ? 'settings.language.keepOne' : 'settings.language.remove', order.length === 1, () =>
        apply(order.filter((_item, position) => position !== index)),
      ),
    ]);
  }

  function addRow() {
    const missing = installedPacks(state).filter((pack) => !order.includes(pack.code));
    if (!missing.length) return h('span', { class: 'setting__note', text: t('settings.language.allAdded') });
    let choice = missing[0].code;
    const select = selectControl(
      choice,
      missing.map((pack) => ({ value: pack.code, label: packLabel(state, pack.code, opts.showCodes) })),
      (value) => {
        choice = value;
      },
    );
    return h('div', { class: 'row' }, [
      select,
      h('button', {
        class: 'button button--secondary button--sm',
        type: 'button',
        text: t('settings.language.addToList'),
        onclick: () => apply([...order, choice]),
      }),
    ]);
  }

  draw();
  return root;
}

function orderButton(glyph, titleKey, disabled, onClick) {
  const label = t(titleKey);
  const button = h('button', {
    class: 'button button--ghost button--sm',
    type: 'button',
    text: glyph,
    title: label,
    'aria-label': label,
    onclick: onClick,
  });
  button.disabled = disabled;
  return button;
}

/* Um pacote habilitado que sumiu do disco continua na lista para poder ser removido:
   escondê-lo deixaria a configuração inválida sem nada na tela explicando por quê. */
function enabledOrder(state) {
  const languages = state.languages || {};
  const enabled = [...(languages.enabled || [])];
  if (!enabled.length && languages.primary) enabled.push(languages.primary);
  return enabled;
}

function installedPacks(state) {
  return ((state.languages || {}).available || []).filter((pack) => pack.valid !== false);
}

function packLabel(state, code, showCode) {
  const pack = ((state.languages || {}).available || []).find((entry) => entry.code === code);
  const name = (pack && pack.name) || code;
  if (name === code) return code;
  return showCode ? `${name} (${code})` : name;
}

function readLanguageOrder(root) {
  const node = root.querySelector('[data-lang-order]');
  return (node.dataset.langOrder || '').split(' ').filter(Boolean);
}

function appearanceCard(state) {
  const themes = [
    { value: 'auto', label: t('settings.appearance.themeAuto') },
    { value: 'light', label: t('settings.appearance.themeLight') },
    { value: 'dark', label: t('settings.appearance.themeDark') },
  ];
  return card('settings.appearance.title', [
    settingRow(t('settings.appearance.theme'), [
      selectControl((state.config.ui || {}).theme, themes, (value) => {
        applyTheme(value);
        save({ ui: { theme: value } }, 'settings.appearance.title');
      }),
    ]),
  ]);
}

function assistantCard() {
  return card('settings.assistant.title', [
    h('p', { class: 'text-muted text-sm', text: t('settings.assistant.help') }),
    h('div', { id: 'llm-body', class: 'stack' }, [empty(t('settings.assistant.detecting'))]),
  ]);
}

/* ---------- modelo de reconhecimento (whisper.cpp) ---------- */

function sttModelCard() {
  return card('settings.sttModel.title', [
    h('p', { class: 'text-muted text-sm', text: t('settings.sttModel.help') }),
    h('div', { id: 'whisper-body', class: 'stack' }, [empty(t('settings.sttModel.loading'))]),
  ]);
}

/* A opção só é gravada quando o build com GPU termina de instalar: gravá-la antes deixaria
   o mod pedindo uma placa que ainda não tem com que rodar, e o arranque seguinte cairia de
   volta na CPU sem ninguém entender por quê.

   Quem lembra do pedido é `gpuWanted`, não a transição do progresso: uma instalação que
   termina entre o POST e a atualização seguinte não tem transição para observar, e o
   `done` sozinho não serve — ele continua verdadeiro depois de a pessoa voltar para a CPU
   de propósito, e voltaria a ligar a GPU sem ninguém ter pedido. */
async function refreshWhisper() {
  const previous = whisperStatus && whisperStatus.download;
  try {
    whisperStatus = await api('/api/stt/models');
  } catch (error) {
    whisperStatus = { models: [], error: error.message, download: {}, gpu: {} };
  }
  const download = whisperStatus.download || {};
  if (previous && previous.active && download.done) {
    toast(t('settings.sttModel.downloaded', { model: download.model }), 'ok');
  }
  const install = (whisperStatus.gpu || {}).install || {};
  if (gpuWanted && !install.active) {
    gpuWanted = false;
    if (install.done) {
      toast(t('settings.sttModel.gpuInstalled'), 'ok');
      await save({ stt: { whisper_cpp: { use_gpu: true } } }, 'settings.sttModel.title');
      whisperStatus.device = 'gpu';
    }
  }
  renderWhisper();
}

function renderWhisper() {
  const body = $('whisper-body');
  if (!body || !whisperStatus) return;
  if (whisperStatus.error) {
    replace(body, [empty(whisperStatus.error)]);
    return;
  }
  const rows = [];
  /* O peso só entra em jogo com o whisper.cpp: dizer isso evita a pessoa baixar um
     modelo grande e não ver diferença nenhuma. */
  if (whisperStatus.engine && whisperStatus.engine !== 'whisper-cpp') {
    rows.push(h('span', { class: 'setting__note', text: t('settings.sttModel.engineNote', { engine: whisperStatus.engine }) }));
  }
  if (whisperStatus.current_file && !whisperStatus.current_known) {
    rows.push(h('span', { class: 'setting__note', text: t('settings.sttModel.custom', { file: whisperStatus.current_file }) }));
  }
  rows.push(...whisperDeviceRows());
  rows.push(h('ul', { class: 'list' }, (whisperStatus.models || []).map(whisperRow)));
  rows.push(...whisperDownloadRows());
  rows.push(
    h('div', { class: 'row' }, [
      h('span', { class: 'setting__note', text: t('settings.sttModel.folder', { path: whisperStatus.directory || '—' }) }),
      h('button', { class: 'button button--ghost button--sm', type: 'button', text: t('settings.sttModel.refresh'), onclick: refreshWhisper }),
    ]),
  );
  replace(body, rows);
  scheduleWhisperPoll();
}

/* A lista já vai do mais leve ao mais preciso, e o tamanho ao lado de cada um diz o
   preço: um selo repetindo isso em palavras só somava ruído em cada linha. */
function whisperRow(model) {
  const busy = whisperBusy();
  return h('li', { class: 'list__item list__item--model' }, [
    h('span', { class: 'mono', text: model.name }),
    model.recommended ? pill(t('settings.sttModel.recommended'), 'pill--ok') : null,
    h('span', { class: 'text-muted text-sm', text: fmtBytes(model.size_bytes) }),
    whisperAction(model, busy),
    model.installed && !model.current ? whisperRemoveButton(model, busy) : null,
  ]);
}

function whisperAction(model, busy) {
  if (model.current) return pill(t('settings.sttModel.inUse'), 'pill--ok');
  const installed = model.installed;
  const button = h('button', {
    class: `button button--${installed ? 'secondary' : 'primary'} button--sm`,
    type: 'button',
    text: t(installed ? 'settings.sttModel.use' : 'settings.sttModel.download'),
    onclick: () => (installed ? useWhisperModel(model) : startWhisperDownload(model)),
  });
  button.disabled = busy;
  return button;
}

function whisperRemoveButton(model, busy) {
  const button = h('button', {
    class: 'button button--ghost button--sm',
    type: 'button',
    text: t('settings.sttModel.remove'),
    onclick: () => removeWhisperModel(model, button),
  });
  button.disabled = busy;
  return button;
}

function whisperDownloadRows() {
  const download = whisperStatus.download || {};
  if (download.active) {
    return [
      h('div', { class: 'row' }, [
        h('span', { class: 'setting__note', text: t('settings.sttModel.downloading', { model: download.model, percent: Math.round(download.percent || 0) }) }),
        h('button', { class: 'button button--ghost button--sm', type: 'button', text: t('settings.sttModel.cancel'), onclick: cancelWhisperDownload }),
      ]),
      progress((download.percent || 0) / 100, !download.percent),
    ];
  }
  if (download.error) {
    return [h('span', { class: 'field__error', text: t('settings.sttModel.downloadFailed', { model: download.model, error: download.error }) })];
  }
  return [];
}

/* Trocar o peso mexe em `stt`, que só é montado no arranque: o próprio save avisa que
   precisa reiniciar, e o selo do cabeçalho fica aceso até isso acontecer.

   A janela do encoder vai junto porque ela é do peso, não do gosto de quem escolhe: o
   corte que acelera o small faz um modelo grande repetir a mesma palavra em laço. */
async function useWhisperModel(model) {
  const patch = { stt: { whisper_cpp: { model: model.config_value, audio_ctx: model.audio_ctx } } };
  if (await save(patch, 'settings.sttModel.title')) refreshWhisper();
}

async function startWhisperDownload(model) {
  try {
    await api('/api/stt/models/download', { method: 'POST', body: { model: model.name } });
    refreshWhisper();
  } catch (error) {
    toast(t('settings.sttModel.downloadFailed', { model: model.name, error: error.message }), 'error');
  }
}

async function cancelWhisperDownload() {
  try {
    await api('/api/stt/models/download/cancel', { method: 'POST' });
  } catch (_error) {
    /* cancelar é melhor esforço: o estado real chega no próximo status */
  }
  refreshWhisper();
}

async function removeWhisperModel(model, button) {
  if (!window.confirm(t('settings.sttModel.removeConfirm', { model: model.name }))) return;
  button.disabled = true;
  try {
    await api('/api/stt/models/delete', { method: 'POST', body: { model: model.name } });
    toast(t('settings.sttModel.removed', { model: model.name }), 'ok');
    refreshWhisper();
  } catch (error) {
    toast(t('settings.sttModel.removeFailed', { model: model.name, error: error.message }), 'error');
    button.disabled = false;
  }
}

/* ---------- onde o whisper.cpp roda (CPU/GPU) ---------- */

/* A GPU só fica clicável quando existe placa que a atenda: o único build com GPU que o
   whisper.cpp publica é o cuBLAS, ou seja, NVIDIA. Oferecer a opção a uma placa AMD seria
   prometer uma troca que o servidor engole em silêncio — ele sobe igual, na CPU, e a
   pessoa fica procurando no lugar errado por que nada acelerou. */
function whisperDeviceRows() {
  const gpu = whisperStatus.gpu || {};
  const device = whisperStatus.device === 'gpu' ? 'gpu' : 'cpu';
  const options = [
    { value: 'cpu', label: t('settings.sttModel.deviceCpu') },
    { value: 'gpu', label: t('settings.sttModel.deviceGpu'), disabled: !gpu.supported },
  ];
  const select = selectControl(device, options, chooseWhisperDevice);
  select.disabled = whisperBusy();
  return [
    settingRow(t('settings.sttModel.device'), [select], whisperDeviceNote(gpu, device)),
    ...whisperGpuRows(gpu, device),
  ];
}

function whisperDeviceNote(gpu, device) {
  if (!gpu.supported) {
    if (gpu.reason === 'old-driver') {
      return t('settings.sttModel.gpuDriverOld', {
        adapter: gpu.adapter,
        driver: gpu.driver,
        required: gpu.required_driver,
      });
    }
    return t('settings.sttModel.gpuNoNvidia');
  }
  if (!gpu.installed) return t('settings.sttModel.gpuNotInstalled', { adapter: gpu.adapter, size: fmtBytes(gpu.download_bytes) });
  return t(device === 'gpu' ? 'settings.sttModel.gpuInUse' : 'settings.sttModel.gpuReady', { adapter: gpu.adapter });
}

/* O erro fica na tela E o botão continua ali: um download cancelado no meio deixa a falha
   registrada, e sem o botão a única saída seria recarregar a página. */
function whisperGpuRows(gpu, device) {
  const install = gpu.install || {};
  if (install.active) {
    return [
      h('div', { class: 'row' }, [
        h('span', {
          class: 'setting__note',
          text: t(install.phase === 'extract' ? 'settings.sttModel.gpuExtracting' : 'settings.sttModel.gpuDownloading', {
            percent: Math.round(install.percent || 0),
          }),
        }),
        h('button', {
          class: 'button button--ghost button--sm',
          type: 'button',
          text: t('settings.sttModel.cancel'),
          onclick: cancelWhisperGpuInstall,
        }),
      ]),
      progress((install.percent || 0) / 100, !install.percent),
    ];
  }
  const rows = [];
  if (install.error) rows.push(h('span', { class: 'field__error', text: t('settings.sttModel.gpuFailed', { error: install.error }) }));
  if (!gpu.supported) return rows;
  if (!gpu.installed) {
    rows.push(
      h('div', { class: 'row' }, [
        h('button', {
          class: 'button button--secondary button--sm',
          type: 'button',
          text: t('settings.sttModel.gpuInstall', { size: fmtBytes(gpu.download_bytes) }),
          onclick: () => startWhisperGpuInstall(gpu),
        }),
      ]),
    );
    return rows;
  }
  if (device === 'gpu') return rows;
  rows.push(
    h('div', { class: 'row' }, [
      h('span', { class: 'setting__note', text: t('settings.sttModel.gpuFolder', { path: gpu.directory, size: fmtBytes(gpu.disk_bytes) }) }),
      h('button', {
        class: 'button button--ghost button--sm',
        type: 'button',
        text: t('settings.sttModel.gpuRemove'),
        onclick: (event) => removeWhisperGpu(event.target),
      }),
    ]),
  );
  return rows;
}

/* Escolher a GPU sem o build instalado não grava nada — dispara a instalação. Gravar antes
   deixaria a configuração pedindo uma placa sem ter com que rodar nela, e o mod voltaria
   para a CPU no arranque seguinte; quem grava é o fim da instalação, em `refreshWhisper`. */
async function chooseWhisperDevice(value) {
  const gpu = whisperStatus.gpu || {};
  if (value === 'gpu' && !gpu.installed) {
    if (!(await startWhisperGpuInstall(gpu))) renderWhisper();
    return;
  }
  if (await save({ stt: { whisper_cpp: { use_gpu: value === 'gpu' } } }, 'settings.sttModel.title')) refreshWhisper();
}

async function startWhisperGpuInstall(gpu) {
  const confirmed = window.confirm(
    t('settings.sttModel.gpuConfirm', {
      build: gpu.build,
      download: fmtBytes(gpu.download_bytes),
      disk: fmtBytes(gpu.disk_bytes),
    }),
  );
  if (!confirmed) return false;
  try {
    await api('/api/stt/gpu/install', { method: 'POST' });
  } catch (error) {
    toast(t('settings.sttModel.gpuFailed', { error: error.message }), 'error');
    return false;
  }
  gpuWanted = true;
  refreshWhisper();
  return true;
}

async function cancelWhisperGpuInstall() {
  gpuWanted = false;
  try {
    await api('/api/stt/gpu/install/cancel', { method: 'POST' });
  } catch (_error) {
    /* cancelar é melhor esforço: o estado real chega no próximo status */
  }
  refreshWhisper();
}

async function removeWhisperGpu(button) {
  const gpu = whisperStatus.gpu || {};
  if (!window.confirm(t('settings.sttModel.gpuRemoveConfirm', { size: fmtBytes(gpu.disk_bytes) }))) return;
  button.disabled = true;
  try {
    await api('/api/stt/gpu/remove', { method: 'POST' });
    toast(t('settings.sttModel.gpuRemoved'), 'ok');
    refreshWhisper();
  } catch (error) {
    toast(t('settings.sttModel.gpuRemoveFailed', { error: error.message }), 'error');
    button.disabled = false;
  }
}

/* Um download por vez: os dois puxam centenas de MB da mesma conexão, e o do build com GPU
   passa de meio giga — deixar os dois correndo juntos só faz cada um demorar o dobro. */
function whisperBusy() {
  const install = (whisperStatus.gpu || {}).install || {};
  return Boolean((whisperStatus.download || {}).active || install.active);
}

function scheduleWhisperPoll() {
  clearTimeout(whisperTimer);
  if (whisperStatus && whisperBusy()) whisperTimer = setTimeout(refreshWhisper, LLM_POLL_MS);
}

function updatesCard(state) {
  const update = updateInfo(state);
  const check = h('button', {
    class: 'button button--secondary',
    type: 'button',
    text: t('update.check'),
    onclick: checkUpdate,
  });
  check.disabled = Boolean(update.busy);
  const rows = [
    settingRow(t('update.current'), [h('span', { class: 'mono', text: update.current || '—' }), check], updateNote(update)),
  ];
  if (update.available) {
    rows.push(
      settingRow(t('update.availableNote', { latest: update.latest }), [
        h('button', { class: 'button button--primary', type: 'button', text: t('update.open'), onclick: openUpdate }),
      ]),
    );
  }
  return card('update.section', rows);
}

function updateNote(update) {
  if (update.error) return update.error;
  if (!update.supported) return t('update.dev');
  if (update.busy) return t(`update.phase.${update.phase}`, { percent: Math.round(update.percent || 0) });
  if (update.available) return update.skipped === update.latest ? t('update.skipped', { version: update.skipped }) : '';
  return update.checked ? t('update.upToDate') : '';
}

/* ---------- assistente (Ollama) ---------- */

async function refreshLlm() {
  const previous = llmStatus && llmStatus.pull;
  try {
    llmStatus = await api('/api/llm');
  } catch (error) {
    llmStatus = { managed: false, error: error.message };
  }
  const pull = llmStatus.pull;
  if (previous && previous.active && pull && pull.done) toast(t('settings.assistant.pulled', { model: pull.model }), 'ok');
  renderLlm();
}

function renderLlm() {
  const body = $('llm-body');
  if (!body || !llmStatus) return;
  const enabled = Boolean(llmStatus.enabled);
  const rows = [
    settingRow(t('settings.assistant.enabled'), [
      toggleControl(enabled, (value) => save({ intent: { llm: { enabled: value } } }, 'settings.assistant.title')),
    ]),
    modeRow(),
  ];
  if (llmStatus.managed) rows.push(...ollamaRows());
  else rows.push(h('span', { class: 'setting__note', text: llmStatus.error || llmStatus.provider || '' }));
  replace(body, rows);
  scheduleLlmPoll();
}

/* A escolha vem da configuração, não do status do Ollama: ela continua valendo com o
   assistente fora do ar, e é a configuração que a interface grava. */
function modeRow() {
  const modes = [
    { value: 'fallback', label: t('settings.assistant.modeFallback') },
    { value: 'pair', label: t('settings.assistant.modePair') },
  ];
  const current = getPath(store.state.config, 'intent.llm.mode') || modes[0].value;
  return settingRow(
    t('settings.assistant.mode'),
    [selectControl(current, modes, (value) => save({ intent: { llm: { mode: value } } }, 'settings.assistant.title'))],
    t('settings.assistant.modeNote'),
  );
}

/* Baixar modelo e listar modelos são chamadas ao servidor: com ele fora do ar não
   há o que oferecer além de dizer para abri-lo. */
function ollamaRows() {
  const rows = [h('div', { class: 'row' }, [ollamaPill(), refreshButton()])];
  if (!llmStatus.installed) {
    rows.push(h('a', { class: 'button button--secondary', href: llmStatus.install_url, target: '_blank', rel: 'noreferrer', text: t('settings.assistant.installLink') }));
    return rows;
  }
  if (llmStatus.binary) {
    rows.push(h('span', { class: 'setting__note', text: t('settings.assistant.found', { path: llmStatus.binary }) }));
  }
  if (!llmStatus.online) {
    rows.push(h('span', { class: 'setting__note', text: t('settings.assistant.startHint') }));
    return rows;
  }
  rows.push(modelList());
  rows.push(pullRow());
  return rows;
}

function modelList() {
  const models = llmStatus.models || [];
  if (!models.length) {
    return h('div', { class: 'stack' }, [
      empty(t('settings.assistant.modelNone')),
      h('span', { class: 'setting__note', text: t('settings.assistant.suggestion', { model: llmStatus.suggested_model }) }),
    ]);
  }
  return h('div', { class: 'stack' }, [
    h('span', { class: 'setting__label', text: t('settings.assistant.installed', { count: models.length }) }),
    h('ul', { class: 'list' }, models.map(modelRow)),
  ]);
}

function modelRow(model) {
  const inUse = model.name === llmStatus.model;
  return h('li', { class: 'list__item list__item--model' }, [
    h('span', { class: 'mono', text: model.name }),
    h('span', { class: 'text-muted text-sm', text: modelSummary(model) }),
    inUse
      ? pill(t('settings.assistant.inUse'), 'pill--ok')
      : h('button', {
          class: 'button button--secondary button--sm',
          type: 'button',
          text: t('settings.assistant.use'),
          onclick: () => save({ intent: { llm: { model: model.name } } }, 'settings.assistant.title'),
        }),
    h('button', {
      class: 'button button--ghost button--sm',
      type: 'button',
      text: t('settings.assistant.remove'),
      onclick: (event) => removeModel(model, inUse, event.target),
    }),
  ]);
}

function modelSummary(model) {
  return [model.parameter_size, fmtBytes(model.size)].filter(Boolean).join(' · ');
}

/* Remover o modelo em uso deixaria o assistente ligado e quebrado: a interface
   troca a seleção junto, ou a esvazia se não sobrar nenhum. */
async function removeModel(model, inUse, button) {
  if (!window.confirm(t('settings.assistant.removeConfirm', { model: model.name }))) return;
  button.disabled = true;
  try {
    await api('/api/llm/delete', { method: 'POST', body: { model: model.name } });
    if (inUse) {
      const replacement = (llmStatus.models || []).find((other) => other.name !== model.name);
      await save({ intent: { llm: { model: replacement ? replacement.name : '' } } }, 'settings.assistant.title');
    }
    toast(t('settings.assistant.removed', { model: model.name }), 'ok');
    refreshLlm();
  } catch (error) {
    toast(t('settings.assistant.removeFailed', { model: model.name, error: error.message }), 'error');
    button.disabled = false;
  }
}

function ollamaPill() {
  if (!llmStatus.installed) return pill(t('settings.assistant.missing'), 'pill--warning');
  if (!llmStatus.online) return pill(t('settings.assistant.offline'), 'pill--warning');
  return pill(t('settings.assistant.online', { endpoint: llmStatus.endpoint }), 'pill--ok');
}

function refreshButton() {
  return h('button', {
    class: 'button button--ghost button--sm',
    type: 'button',
    text: t('settings.assistant.refresh'),
    onclick: refreshLlm,
  });
}

/* Um <select> de verdade em vez de <datalist>: o datalist filtra pelo que já foi
   digitado, então uma letra que não casa com sugestão nenhuma abre uma lista vazia
   e parece quebrado. Aqui as opções estão sempre à vista, e "Outro…" abre o campo
   livre para quem já sabe o nome do modelo que quer. */
function pullRow() {
  const pull = llmStatus.pull || {};
  const installed = (llmStatus.models || []).map((model) => model.name);
  const choices = (llmStatus.suggested_models || [])
    .filter((name) => !installed.includes(name))
    .map((name) => ({ value: name, label: name }));
  choices.push({ value: OTHER_MODEL, label: t('settings.assistant.otherModel') });
  if (!choices.some((choice) => choice.value === pullChoice)) pullChoice = choices[0].value;

  const select = selectControl(pullChoice, choices, (value) => {
    pullChoice = value;
    renderLlm();
  });
  const custom = h('input', {
    class: 'field__control',
    type: 'text',
    placeholder: t('settings.assistant.modelPlaceholder'),
    autocomplete: 'off',
    oninput: (event) => {
      pullCustom = event.target.value;
    },
  });
  custom.value = pullCustom;
  const wanted = pullChoice === OTHER_MODEL ? pullCustom : pullChoice;
  const button = h('button', {
    class: 'button button--primary',
    type: 'button',
    text: t('settings.assistant.pull'),
    onclick: () => startPull(wanted),
  });
  button.disabled = Boolean(pull.active);
  const controls = pullChoice === OTHER_MODEL ? [select, custom, button] : [select, button];
  const rows = [settingRow(t('settings.assistant.pull'), controls)];
  if (pull.active) {
    rows.push(h('span', { class: 'setting__note', text: t('settings.assistant.pulling', { model: pull.model, percent: pull.percent }) }));
    rows.push(progress(pull.percent / 100, !pull.percent));
  } else if (pull.error) {
    rows.push(h('span', { class: 'field__error', text: t('settings.assistant.pullFailed', { model: pull.model, error: pull.error }) }));
  }
  return h('div', { class: 'stack' }, rows);
}

async function startPull(model) {
  try {
    await api('/api/llm/pull', { method: 'POST', body: { model } });
    refreshLlm();
  } catch (error) {
    toast(t('settings.assistant.pullFailed', { model, error: error.message }), 'error');
  }
}

function scheduleLlmPoll() {
  clearTimeout(llmTimer);
  if (llmStatus && llmStatus.pull && llmStatus.pull.active) llmTimer = setTimeout(refreshLlm, LLM_POLL_MS);
}

/* ---------- registro de idiomas ---------- */

async function loadRegistry() {
  registry = { entries: [], error: '', loading: true };
  renderRegistry();
  try {
    registry = { ...(await api('/api/languages/registry')), loading: false };
  } catch (error) {
    registry = { entries: [], error: error.message, loading: false };
  }
  renderRegistry();
}

function renderRegistry() {
  const body = $('registry-body');
  if (!body) return;
  if (registry === null) {
    loadRegistry();
    return;
  }
  if (registry.loading) {
    replace(body, empty(t('settings.language.registryLoading')));
    return;
  }
  if (registry.error) {
    replace(body, empty(t('settings.language.registryFailed', { error: registry.error })));
    return;
  }
  const needle = registryQuery.trim().toLowerCase();
  const matches = registry.entries.filter(
    (entry) => !entry.installed && (!needle || `${entry.code} ${entry.name}`.toLowerCase().includes(needle)),
  );
  if (!matches.length) {
    replace(body, empty(t(needle ? 'settings.language.noMatch' : 'settings.language.registryEmpty')));
    return;
  }
  replace(body, matches.map(registryRow));
}

function registryRow(entry) {
  const button = h('button', {
    class: 'button button--secondary button--sm',
    type: 'button',
    text: t('settings.language.install'),
    onclick: () => installPack(entry, button),
  });
  return h('div', { class: 'setting' }, [
    h('span', { class: 'setting__label', text: `${entry.name} (${entry.code})` }),
    h('div', { class: 'setting__control' }, [button]),
    entry.description ? h('span', { class: 'setting__note', text: entry.description }) : null,
  ]);
}

async function installPack(entry, button) {
  button.disabled = true;
  button.textContent = t('settings.language.installing');
  try {
    const response = await api('/api/languages/install', { method: 'POST', body: { code: entry.code } });
    store.state.languages = response.languages;
    registry = null;
    store.settingsVersion += 1;
    toast(t('settings.language.installed', { name: entry.name }), 'ok');
    scheduleRender();
  } catch (error) {
    toast(t('settings.language.installFailed', { name: entry.name, error: error.message }), 'error');
    button.disabled = false;
    button.textContent = t('settings.language.install');
  }
}

/* ---------- gravação ---------- */

/* `languages` é derivado da configuração no servidor. Sem espelhar a gravação aqui, o
   re-render logo depois de salvar redesenha a lista com a ordem antiga, e ela só volta
   ao certo quando o snapshot chega — pisca como se a mexida tivesse sido descartada. */
function syncLanguages(config) {
  const languages = store.state.languages;
  if (!languages) return;
  const saved = config.languages || {};
  languages.enabled = [...(saved.enabled || [])];
  languages.primary = saved.primary;
}

async function save(patch, sectionKey) {
  try {
    const response = await api('/api/config', { method: 'PUT', body: { patch } });
    store.state.config = response.config;
    syncLanguages(response.config);
    store.state.restart_pending = store.state.restart_pending || response.restart_required;
    store.settingsVersion += 1;
    applyTheme(response.config.ui.theme);
    toast(
      t(response.restart_required ? 'settings.savedRestart' : 'settings.saved', { section: t(sectionKey) }),
      response.restart_required ? 'warning' : 'ok',
    );
    scheduleRender();
    return true;
  } catch (error) {
    toast(t('settings.error', { section: t(sectionKey), error: error.message }), 'error');
    return false;
  }
}

async function refreshDevices() {
  try {
    store.state.devices = await api('/api/devices?refresh=1');
    store.settingsVersion += 1;
    scheduleRender();
  } catch (_error) {
    store.state.devices = [];
  }
}

/* ---------- avançado ---------- */

function advancedCard(state, section) {
  const form = h('form', { class: 'form', 'data-section': section.id, novalidate: true });
  if (section.languages) languageFields(state, form);
  else section.paths.forEach((path) => schemaFields(state, form, section, path));
  const errorBox = h('div', { class: 'field__error', role: 'alert' });
  form.appendChild(
    h('div', { class: 'form__footer' }, [
      h('button', { class: 'button button--primary', type: 'submit', text: t('settings.save') }),
      sectionRequiresRestart(section) ? pill(t('settings.advanced.restartNeeded'), 'pill--warning') : null,
      errorBox,
    ]),
  );
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    saveSection(section, form, errorBox);
  });
  return h('section', { class: 'card grid__span-2' }, [
    h('h2', { class: 'card__title', text: t(section.title) }),
    h('div', { class: 'card__body' }, [form]),
  ]);
}

function sectionRequiresRestart(section) {
  return section.paths.some((path) => RESTART_SECTIONS.has(path.split('.')[0]));
}

function schemaFields(state, form, section, path) {
  const fields = (state.config_schema || {})[path] || {};
  Object.entries(fields).forEach(([key, spec]) => {
    const dotted = `${path}.${key}`;
    const label = section.paths.length > 1 ? dotted : key;
    const value = getPath(state.config, dotted);
    const resolved = dotted === DEVICE_PATH ? withDeviceChoices(spec, state) : spec;
    form.appendChild(fieldControl(dotted, label, resolved, value));
  });
}

function withDeviceChoices(spec, state) {
  const choices = [{ value: SYSTEM_DEFAULT_DEVICE, text: t('settings.mic.deviceDefault') }];
  (state.devices || []).forEach((device) => choices.push({ value: device.name, text: device.name }));
  return Object.assign({}, spec, { choices });
}

function fieldControl(dotted, label, spec, value) {
  const id = `cfg-${dotted.replace(/\./g, '-')}`;
  const control = buildControl(id, dotted, spec, value);
  if (spec.type === 'boolean') {
    return h('div', { class: 'field' }, [h('label', { class: 'toggle', for: id }, [control, h('span', { class: 'field__label', text: label })])]);
  }
  return h('div', { class: 'field' }, [
    h('label', { class: 'field__label', for: id, text: label }),
    control,
    h('span', { class: 'field__help', text: fieldHelp(spec) }),
  ]);
}

function buildControl(id, dotted, spec, value) {
  const base = { class: 'field__control', id, 'data-path': dotted };
  if (spec.type === 'boolean') {
    const input = h('input', { class: 'toggle__input', type: 'checkbox', id, 'data-path': dotted, 'data-type': 'boolean' });
    input.checked = Boolean(value);
    return input;
  }
  if (spec.choices || spec.enum) {
    const options = spec.choices
      ? spec.choices.map((choice) => h('option', { value: choice.value, text: choice.text }))
      : spec.enum.map((option) => h('option', { value: option, text: String(option) }));
    const node = h('select', { ...base, 'data-type': spec.choices ? 'string' : spec.type }, options);
    const current = value === null || value === undefined ? '' : String(value);
    if (current && !Array.from(node.options).some((option) => option.value === current)) {
      node.appendChild(h('option', { value: current, text: current }));
    }
    node.value = current;
    return node;
  }
  if (spec.type === 'integer' || spec.type === 'number') {
    const node = h('input', {
      ...base,
      type: 'number',
      'data-type': spec.type,
      min: spec.min,
      max: spec.max,
      step: spec.type === 'integer' ? 1 : 'any',
    });
    node.value = value === null || value === undefined ? '' : String(value);
    return node;
  }
  if (spec.type === 'array') {
    const node = h('input', { ...base, type: 'text', 'data-type': 'array' });
    node.value = Array.isArray(value) ? value.join(', ') : '';
    return node;
  }
  const node = h('input', { ...base, type: 'text', 'data-type': 'string' });
  node.value = value === null || value === undefined ? '' : String(value);
  return node;
}

function fieldHelp(spec) {
  const parts = [];
  if (spec.min !== undefined && spec.max !== undefined) parts.push(t('settings.advanced.range', { min: spec.min, max: spec.max }));
  else if (spec.min !== undefined) parts.push(t('settings.advanced.rangeMin', { min: spec.min }));
  else if (spec.max !== undefined) parts.push(t('settings.advanced.rangeMax', { max: spec.max }));
  parts.push(t('settings.advanced.defaultValue', { value: JSON.stringify(spec.default) }));
  return parts.join(' · ');
}

function readControl(control) {
  const type = control.dataset.type;
  if (type === 'boolean') return control.checked;
  if (type === 'integer') return control.value === '' ? null : parseInt(control.value, 10);
  if (type === 'number') return control.value === '' ? null : parseFloat(control.value);
  if (type === 'array') return control.value.split(',').map((item) => item.trim()).filter(Boolean);
  return control.value;
}

/* O mesmo widget do básico, com os códigos à mostra: aqui o código É a informação
   útil. Some o par de campos separados — a ordem já responde `enabled` e `primary`,
   então o estado inválido que a validação avisava deixou de existir. */
/* Os dois campos ocupam a linha inteira: espremidos lado a lado, o nome do idioma fica
   com menos de um terço da coluna e o erro do índice vira uma torre de dez linhas. */
function languageFields(state, form) {
  form.appendChild(
    h('div', { class: 'field field--wide' }, [
      h('span', { class: 'field__label', text: 'enabled · primary' }),
      languageOrderControl(state, { showCodes: true }),
      h('span', { class: 'field__help', text: t('settings.language.orderHelp') }),
    ]),
  );
  form.appendChild(h('div', { class: 'field field--wide' }, [registrySearch()]));
}

function collectPatch(section, form) {
  const patch = {};
  const current = store.state.config;
  if (section.languages) {
    const enabled = readLanguageOrder(form);
    const languages = current.languages || {};
    if (JSON.stringify(enabled) !== JSON.stringify(languages.enabled)) setPath(patch, 'languages.enabled', enabled);
    if (enabled[0] !== languages.primary) setPath(patch, 'languages.primary', enabled[0]);
    return patch;
  }
  form.querySelectorAll('[data-path]').forEach((control) => {
    const value = readControl(control);
    if (JSON.stringify(value) !== JSON.stringify(getPath(current, control.dataset.path))) {
      setPath(patch, control.dataset.path, value);
    }
  });
  return patch;
}

function patchRequiresRestart(patch) {
  if (Object.keys(patch).some((section) => RESTART_SECTIONS.has(section))) return true;
  return Boolean(patch.intent) && Object.keys(patch.intent).some((key) => RESTART_INTENT_FIELDS.has(key));
}

async function saveSection(section, form, errorBox) {
  errorBox.textContent = '';
  const patch = collectPatch(section, form);
  const title = t(section.title);
  if (!Object.keys(patch).length) {
    toast(t('settings.nothingToSave', { section: title }), 'warning');
    return;
  }
  try {
    const response = await api('/api/config', { method: 'PUT', body: { patch } });
    const restart = response.restart_required || patchRequiresRestart(patch);
    store.state.config = response.config;
    syncLanguages(response.config);
    store.state.restart_pending = store.state.restart_pending || restart;
    store.settingsVersion += 1;
    applyTheme(response.config.ui.theme);
    toast(t(restart ? 'settings.savedRestart' : 'settings.saved', { section: title }), restart ? 'warning' : 'ok');
    scheduleRender();
  } catch (error) {
    errorBox.textContent = error.message;
    toast(t('settings.error', { section: title, error: error.message }), 'error');
  }
}
