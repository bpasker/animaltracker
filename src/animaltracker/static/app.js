/* ============================================================================
   app.js — boot, theme, the application shell, and the route table.

   WHAT THIS FILE OWNS (so a view never has to)
     · the four-surface navigation: desktop .appbar, mobile .topbar + .tabbar
     · the desktop .rail's fixed furniture — per-camera health rows, saved
       views, and the disk/retention footer. A view contributes only its own
       rail sections (the filter panel) through the chrome slice.
     · the theme control, persisted at localStorage['at:theme'] as a RAW
       string, because the render-blocking script in the shell <head> reads it
       with a bare getItem() before first paint.
     · polling that pauses when the document is hidden.
     · the global keyboard layer: "/" search, Cmd/Ctrl+K palette, "g" then
       l/r/m/s, "?" shortcuts, Escape closes the topmost overlay.
     · --page-pad, the content gutter (see below).

   THE CHROME CONTRACT — how a view talks to the shell
   A view never reaches into the app bar, the rail or the tab bar. It calls
   store.setChrome() and this file renders the result:

     store.setChrome({
       title:    'Recordings',            // mobile top bar + document.title
       subtitle: '250 clips · 11 species',// mono sub-line under the title
       actions:  [buttonEl, ...],         // trailing controls, mobile top bar
       toolbar:  chipStripEl,             // rides under the mobile top bar;
                                          // presence adds .shell--toolbar so
                                          // .dayhead sticks below it
       rail:     railSectionsEl,          // the view's own rail sections; they
                                          // render between camera health and
                                          // the disk footer. null = none.
       norail:   false,                   // true renders no rail at all
       selbar:   selbarEl,                // child of .shell (chrome, not a
                                          // modal); presence adds
                                          // .shell--selbar so .main clears it
       mods:     ['no-blur']              // extra shell modifiers, minus the
                                          // 'shell--' prefix
     })

   Every key is optional; app.js resets the whole slice before mounting a
   view, so a view declares only what it needs and never has to clean up.

   --page-pad — THE CONTENT GUTTER
   app.css deliberately gives .main no horizontal padding (it carries only the
   safe-area insets), and .chip-row's edge-bleed is written as a negative
   margin of --s-4 below 640px and --s-6 above. This file reconciles the two:
   it publishes --page-pad on the shell at the same breakpoint and applies it
   to .main. A view that wants a full-bleed strip uses
   `margin-inline: calc(var(--page-pad) * -1)`; everything else just works.
   ========================================================================= */

import { h, clear, on, keyedList } from './core/dom.js';
import { icon, installSprite } from './core/icons.js';
import { store, readLocal, writeLocal } from './core/store.js';
import { router } from './core/router.js';
import { api } from './core/api.js';
import { toast } from './core/toast.js';
import { sheet, dialog, palette, closeTop, isOverlayOpen } from './core/overlay.js';
import { shortAgo } from './core/format.js';
import { view as recordingsView } from './views/recordings.js';

var THEME_KEY = 'at:theme';
var CAMERA_POLL_MS = 5000;
var MONITOR_POLL_MS = 30000;

var SURFACES = [
  { path: '/live', label: 'Live', icon: 'live' },
  { path: '/recordings', label: 'Recordings', icon: 'film' },
  { path: '/monitor', label: 'Monitor', icon: 'monitor' },
  { path: '/settings', label: 'Settings', icon: 'settings' }
];

/* --------------------------------------------------------------------------
   THEME
   The <head> script has already applied the stored value; this only keeps the
   control, the attribute and storage in agreement afterwards.
   ------------------------------------------------------------------------ */
function readTheme() {
  try {
    var t = window.localStorage.getItem(THEME_KEY);
    return (t === 'light' || t === 'dark') ? t : 'auto';
  } catch (e) { return 'auto'; }
}

function applyTheme(mode) {
  if (mode === 'light' || mode === 'dark') {
    document.documentElement.setAttribute('data-theme', mode);
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  try {
    if (mode === 'auto') window.localStorage.removeItem(THEME_KEY);
    else window.localStorage.setItem(THEME_KEY, mode);
  } catch (e) { /* private window: the choice lasts for this session only */ }
  store.set({ theme: mode });
}

function resolvedTheme(mode) {
  if (mode !== 'auto') return mode;
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  } catch (e) { return 'light'; }
}

/* --------------------------------------------------------------------------
   SMALL SHARED BUILDERS
   ------------------------------------------------------------------------ */
function iconButton(name, label, onClick, extraClass) {
  var btn = h('button.icon-btn', {
    type: 'button',
    'aria-label': label,
    'class': extraClass || null
  }, icon(name));
  if (onClick) btn.addEventListener('click', onClick);
  return btn;
}

function labelledButton(opts) {
  var btn = h('button.btn', {
    type: 'button',
    'class': 'btn--' + (opts.variant || 'secondary')
  });
  if (opts.icon) btn.appendChild(icon(opts.icon, { size: 'sm', 'class': 'btn__icon' }));
  btn.appendChild(h('span.btn__label', { text: opts.label }));
  if (opts.onClick) btn.addEventListener('click', opts.onClick);
  return btn;
}

/* Camera state -> the three redundant channels: dot class, word, sub-line. */
function cameraWords(cam) {
  var state = cam.state || 'unknown';
  if (state === 'live') {
    return { dot: 'live', word: 'Live', meta: cam.id + ' · live' };
  }
  if (state === 'stale') {
    return {
      dot: 'stale', word: 'Stale',
      meta: cam.id + ' · stale ' + shortAgo(cam.frame_age)
    };
  }
  if (state === 'offline') {
    return { dot: 'offline', word: 'Offline', meta: cam.id + ' · no frames' };
  }
  return { dot: 'offline', word: 'Unknown', meta: cam.id };
}

/* ==========================================================================
   THE SHELL
   ========================================================================= */
function buildShell(root) {
  var shell = {};

  /* --- desktop app bar --------------------------------------------------- */
  var brand = h('a.brand', { href: router.href('/recordings'), 'aria-label': 'Animal Tracker, go to Recordings' },
    h('span.brand__mark', icon('camera', { size: 'sm' })),
    h('span.brand__name', 'Animal ', h('span', 'Tracker')));

  var tabs = h('nav.appbar__tabs', { 'aria-label': 'Primary' });
  shell.tabs = [];
  SURFACES.forEach(function (s) {
    var a = h('a.tab', { href: router.href(s.path) },
      icon(s.icon, { size: 'sm' }),
      h('span', { text: s.label }));
    a.dataset.path = s.path;
    tabs.appendChild(a);
    shell.tabs.push(a);
  });

  var searchInput = h('input.search__input', {
    type: 'search',
    id: 'at-global-search',
    autocomplete: 'off',
    autocorrect: 'off',
    spellcheck: 'false',
    placeholder: 'Search species, camera, filename'
  });
  searchInput.setAttribute('aria-label', 'Search the archive');
  var searchClear = iconButton('x', 'Clear search', null, 'search__clear');
  var searchField = h('div.search.appbar__search',
    h('span.search__icon', icon('search', { size: 'sm' })),
    searchInput,
    h('kbd.search__kbd', '/'),
    searchClear);
  shell.searchInput = searchInput;
  shell.searchField = searchField;

  var paletteBtn = h('button.btn.btn--secondary', { type: 'button', 'aria-label': 'Open the command palette' },
    icon('command', { size: 'sm', 'class': 'btn__icon' }),
    h('span.btn__label', 'Actions'),
    h('kbd.kbd', isMac() ? '⌘K' : 'Ctrl K'));
  paletteBtn.addEventListener('click', openPalette);

  shell.themeswitch = buildThemeSwitch();

  var appbar = h('header.appbar', brand, tabs,
    h('span.appbar__spacer'),
    searchField,
    h('div.appbar__end', paletteBtn, shell.themeswitch));

  /* --- mobile top bar ---------------------------------------------------- */
  shell.title = h('h1.topbar__title', { tabIndex: -1 }, 'Animal Tracker');
  shell.subtitle = h('small');
  shell.title.appendChild(shell.subtitle);

  shell.statusBtn = h('button.topbar__status', { type: 'button' },
    h('span.camrow__dot'), h('span', 'Cameras'));
  shell.statusBtn.addEventListener('click', openCameraSheet);

  shell.actions = h('div.topbar__actions');
  var topRow = h('div.topbar__row', shell.title, shell.statusBtn, shell.actions);
  shell.toolbarSlot = h('div.topbar__toolbar');
  shell.toolbarSlot.hidden = true;
  shell.topbar = h('div.topbar', topRow, shell.toolbarSlot);

  /* --- rail -------------------------------------------------------------- */
  shell.railCams = h('div.rail__section');
  shell.railSaved = h('div.rail__section');
  shell.railViewSlot = h('div.rail__section');
  shell.railFoot = h('div.rail__foot');
  shell.rail = h('aside.rail', { 'aria-label': 'Filters and camera health' },
    shell.railCams, shell.railViewSlot, shell.railSaved, shell.railFoot);

  /* --- main -------------------------------------------------------------- */
  shell.main = h('main.main#main', { tabIndex: -1 });
  shell.body = h('div.shell__body', shell.rail, shell.main);

  /* --- mobile tab bar ---------------------------------------------------- */
  var tabbar = h('nav.tabbar', { 'aria-label': 'Primary' });
  shell.tabbarItems = [];
  SURFACES.forEach(function (s) {
    var a = h('a.tabbar__item', { href: router.href(s.path) },
      h('span.tabbar__icon', icon(s.icon)),
      h('span.tabbar__label', { text: s.label }));
    a.dataset.path = s.path;
    tabbar.appendChild(a);
    shell.tabbarItems.push(a);
  });

  clear(root);
  root.appendChild(appbar);
  root.appendChild(shell.topbar);
  root.appendChild(shell.body);
  root.appendChild(tabbar);
  root.removeAttribute('aria-busy');

  /* Polite announcements for route changes and counts. */
  shell.live = h('div.visually-hidden', { role: 'status', 'aria-live': 'polite' });
  document.body.appendChild(shell.live);

  /* Search wiring: typing filters the archive, so it always lands on
     Recordings. Debounced, and it replaces history rather than pushing a
     state per keystroke. */
  var searchTimer = null;
  on(searchInput, 'input', function () {
    searchField.classList.toggle('search--filled', !!searchInput.value);
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = window.setTimeout(function () {
      var q = searchInput.value.trim();
      if (router.current && router.current.path.indexOf('/recordings') === 0) {
        router.setQuery({ q: q || null }, { replace: true });
      } else {
        router.go('/recordings', q ? { q: q } : {});
      }
    }, 260);
  });
  on(searchInput, 'keydown', function (ev) {
    if (ev.key === 'Escape') { searchInput.value = ''; searchInput.blur(); }
  });
  on(searchClear, 'click', function () {
    searchInput.value = '';
    searchField.classList.remove('search--filled');
    searchInput.focus();
    if (router.current && router.current.path.indexOf('/recordings') === 0) {
      router.setQuery({ q: null }, { replace: true });
    }
  });

  return shell;
}

function isMac() {
  return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
}

function buildThemeSwitch() {
  var seg = h('div.seg.themeswitch', { role: 'group', 'aria-label': 'Colour theme' });
  var modes = [
    { value: 'light', label: 'Light', icon: 'sun' },
    { value: 'dark', label: 'Dark', icon: 'moon' },
    { value: 'auto', label: 'Auto', icon: 'auto' }
  ];
  var buttons = [];
  modes.forEach(function (m, i) {
    var btn = h('button.seg__btn', { type: 'button', tabIndex: -1 },
      icon(m.icon, { size: 'sm' }),
      h('span.btn__label', { text: m.label }));
    btn.dataset.mode = m.value;
    btn.addEventListener('click', function () { applyTheme(m.value); });
    buttons.push(btn);
    seg.appendChild(btn);
    if (i === 0) btn.tabIndex = 0;
  });

  /* Roving tabindex: one tab stop for the group, arrows inside. */
  seg.addEventListener('keydown', function (ev) {
    var idx = buttons.indexOf(document.activeElement);
    if (idx < 0) return;
    var next = -1;
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') next = (idx + 1) % buttons.length;
    else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') next = (idx - 1 + buttons.length) % buttons.length;
    else if (ev.key === 'Home') next = 0;
    else if (ev.key === 'End') next = buttons.length - 1;
    if (next < 0) return;
    ev.preventDefault();
    buttons.forEach(function (b, i) { b.tabIndex = i === next ? 0 : -1; });
    buttons[next].focus();
  });

  function sync() {
    var mode = store.get('theme');
    buttons.forEach(function (b) {
      var on_ = b.dataset.mode === mode;
      b.setAttribute('aria-pressed', on_ ? 'true' : 'false');
      var name = b.dataset.mode;
      b.setAttribute('aria-label',
        name === 'auto'
          ? 'Match the system theme, currently ' + resolvedTheme('auto')
          : 'Use the ' + name + ' theme');
    });
  }
  store.select(['theme'], sync);
  sync();
  return seg;
}

/* ==========================================================================
   RAIL FURNITURE — camera health, saved views, disk footer
   ========================================================================= */
function renderRailCameras(shell) {
  var cams = store.get('cameras') || [];
  var err = store.get('camerasError');

  if (!shell.railCams.firstChild || shell.railCams.dataset.built !== '1') {
    clear(shell.railCams);
    shell.railCams.dataset.built = '1';
    shell.railCams.appendChild(h('div.rail__label', 'Cameras',
      h('span.count', { text: String(cams.length || '') })));
    shell.railCams.appendChild(h('div.savedviews#at-rail-cams'));
  }
  var label = shell.railCams.querySelector('.rail__label .count');
  var list = shell.railCams.querySelector('#at-rail-cams');

  if (err) {
    label.textContent = '';
    clear(list).appendChild(h('p.savedviews__empty',
      { text: 'Camera health unavailable — ' + api.describe(err) }));
    return;
  }
  label.textContent = cams.length ? String(cams.length) : '';

  if (!cams.length) {
    clear(list).appendChild(h('p.savedviews__empty',
      { text: 'No cameras are configured. Add one in Settings.' }));
    return;
  }

  keyedList(list, cams, {
    key: function (c) { return c.id; },
    create: function (c) {
      return h('a.camrow', { href: router.href('/live', { camera: c.id }) },
        h('span.camrow__dot'),
        h('span.camrow__name'),
        h('span.camrow__meta'));
    },
    update: function (el, c) {
      var w = cameraWords(c);
      el.className = 'camrow' + (c.state === 'stale' ? ' camrow--stale'
        : (c.state === 'offline' ? ' camrow--offline' : ''));
      var dot = el.querySelector('.camrow__dot');
      dot.className = 'camrow__dot camrow__dot--' + w.dot;
      el.querySelector('.camrow__name').textContent = c.name || c.id;
      el.querySelector('.camrow__meta').textContent = w.meta;
      el.setAttribute('aria-label', (c.name || c.id) + ', ' + w.word + '. Open Live.');
    }
  });
}

function renderRailFoot(shell) {
  var sys = store.get('system');
  clear(shell.railFoot);
  if (!sys) {
    shell.railFoot.appendChild(h('p.savedviews__empty', 'Disk usage loading…'));
    return;
  }
  var pct = Math.max(0, Math.min(100, Number(sys.disk_percent) || 0));
  var lit = Math.round(pct / 100 * 12);
  var word = pct >= 92 ? 'critical' : (pct >= 80 ? 'high' : 'nominal');
  var meter = h('span.meter.meter--wide', {
    'aria-hidden': 'true',
    style: { '--n': String(lit), '--of': '12' }
  });
  if (pct >= 92) meter.classList.add('meter--danger');
  else if (pct >= 80) meter.classList.add('meter--stale');
  else meter.classList.add('meter--accent');

  shell.railFoot.appendChild(h('div.readout',
    h('span.readout__label', 'Disk'),
    h('span.readout__value', { text: Math.round(sys.disk_used_gb) + ' / ' + Math.round(sys.disk_total_gb) },
      h('span.readout__unit', 'GB'))));
  shell.railFoot.appendChild(meter);
  shell.railFoot.appendChild(h('span.visually-hidden',
    { text: 'Disk ' + Math.round(pct) + ' per cent used, ' + word }));
  shell.railFoot.appendChild(h('p.t-micro.t-3',
    { text: Math.round(pct) + '% used · ' + word }));
}

/* --- saved views ---------------------------------------------------------
   Named persistent queries. Stored locally (the server has no endpoint for
   them yet); counts are live, fetched with limit=1 so only `total` matters. */
var SAVED_KEY = 'at:savedviews';

function loadSavedViews() {
  var list = readLocal(SAVED_KEY, []);
  return Array.isArray(list) ? list : [];
}

function currentFilterQuery() {
  var q = (router.current && router.current.query) || {};
  var out = {};
  ['cameras', 'species', 'from', 'to', 'q', 'sort'].forEach(function (k) {
    if (q[k]) out[k] = q[k];
  });
  return out;
}

function sameQuery(a, b) {
  var ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (var i = 0; i < ka.length; i++) if (a[ka[i]] !== b[ka[i]]) return false;
  return true;
}

function renderSavedViews(shell) {
  var views = store.get('savedViews') || [];
  var current = currentFilterQuery();
  var dirty = Object.keys(current).length > 0;

  clear(shell.railSaved);
  shell.railSaved.appendChild(h('div.rail__label', 'Saved views'));
  var list = h('div.savedviews');
  shell.railSaved.appendChild(list);

  if (!views.length) {
    list.appendChild(h('p.savedviews__empty',
      'Filter the archive, then save the view so the same work queue is one click away.'));
  }

  views.forEach(function (v) {
    var active = sameQuery(current, v.query || {});
    var row = h('a.savedview', {
      href: router.href('/recordings', v.query || {}),
      'aria-current': active ? 'true' : null
    },
      icon('bookmark', { size: 'sm' }),
      h('span.savedview__name', { text: v.name }),
      h('span.savedview__count', { text: v.count === undefined ? '·' : String(v.count) }));
    row.addEventListener('contextmenu', function (ev) {
      ev.preventDefault();
      removeSavedView(v.id);
    });
    list.appendChild(row);
  });

  if (dirty && !views.some(function (v) { return sameQuery(current, v.query || {}); })) {
    var save = h('button.savedviews__save', { type: 'button' },
      icon('plus', { size: 'sm' }), h('span', 'Save this view'));
    save.addEventListener('click', function () { promptSaveView(current); });
    shell.railSaved.appendChild(save);
  }
}

function promptSaveView(query) {
  var host = h('div.field');
  var input = h('input.input', { type: 'text', id: 'at-savedview-name',
    placeholder: 'Deer, last 7 days' });
  host.appendChild(h('label.field__label', { for: 'at-savedview-name' }, 'Name this view'));
  host.appendChild(input);

  var dlg = dialog({
    role: 'dialog',
    title: 'Save this view',
    body: 'It appears in the rail with a live count of matching clips.',
    content: host,
    initialFocus: input,
    actions: [
      { label: 'Save', variant: 'primary', value: 'save' },
      { label: 'Cancel', variant: 'secondary', value: null }
    ]
  });
  dlg.result.then(function (value) {
    if (value !== 'save') return;
    var name = input.value.trim();
    if (!name) { toast.error('A saved view needs a name.'); return; }
    var views = loadSavedViews();
    views.push({ id: 'sv' + Date.now(), name: name, query: query });
    writeLocal(SAVED_KEY, views);
    store.set({ savedViews: views });
    refreshSavedCounts();
    toast.success('Saved view “' + name + '”');
  });
}

function removeSavedView(id) {
  var views = loadSavedViews().filter(function (v) { return v.id !== id; });
  var gone = loadSavedViews().filter(function (v) { return v.id === id; })[0];
  writeLocal(SAVED_KEY, views);
  store.set({ savedViews: views });
  toast('Saved view removed', {
    kind: 'danger',
    detail: gone ? gone.name : '',
    undo: {
      onUndo: function () {
        var back = loadSavedViews();
        if (gone) back.push(gone);
        writeLocal(SAVED_KEY, back);
        store.set({ savedViews: back });
        refreshSavedCounts();
      }
    }
  });
}

function refreshSavedCounts() {
  var views = loadSavedViews();
  if (!views.length) return;
  views.forEach(function (v) {
    var q = Object.assign({}, v.query || {}, { limit: 1, offset: 0 });
    if (q.cameras) { q.camera = q.cameras; delete q.cameras; }
    api.recordings(q).then(function (res) {
      var live = loadSavedViews();
      var hit = live.filter(function (x) { return x.id === v.id; })[0];
      if (!hit) return;
      hit.count = res.total;
      writeLocal(SAVED_KEY, live);
      store.set({ savedViews: live.slice() });
    }, function () { /* a count is decoration; the row still works */ });
  });
}

/* ==========================================================================
   CAMERA SHEET (the mobile rail substitute)
   ========================================================================= */
function openCameraSheet() {
  var cams = store.get('cameras') || [];
  var body = h('div.stack');
  if (!cams.length) {
    body.appendChild(h('p.t-sm.t-3',
      'No cameras are configured, so nothing can produce clips. Add one in Settings.'));
  }
  cams.forEach(function (c) {
    var w = cameraWords(c);
    var row = h('a.camrow', { href: router.href('/live', { camera: c.id }) },
      h('span.camrow__dot', { 'class': 'camrow__dot--' + w.dot }),
      h('span.camrow__name', { text: c.name || c.id }),
      h('span.camrow__meta', { text: w.meta + ' · ' + w.word }));
    body.appendChild(row);
  });
  sheet({ title: 'Camera health', snap: 'half', content: body });
}

/* ==========================================================================
   COMMAND PALETTE
   ========================================================================= */
function paletteItems() {
  var items = [];
  SURFACES.forEach(function (s) {
    items.push({
      name: 'Go to ' + s.label, group: 'Navigate', icon: s.icon,
      run: function () { router.go(s.path, {}); }
    });
  });
  items.push({
    name: 'Recordings — Month view', group: 'Navigate', icon: 'calendar',
    run: function () { router.go('/recordings', { view: 'month' }); }
  });
  ['light', 'dark', 'auto'].forEach(function (mode) {
    items.push({
      name: 'Switch to the ' + mode + ' theme', group: 'Appearance',
      icon: mode === 'dark' ? 'moon' : (mode === 'light' ? 'sun' : 'auto'),
      run: function () { applyTheme(mode); }
    });
  });
  (store.get('cameras') || []).forEach(function (c) {
    items.push({
      name: 'Save the last 30 s from ' + (c.name || c.id), group: 'Cameras',
      icon: 'film', scope: c.id,
      run: function () { saveClipFrom(c); }
    });
    items.push({
      name: 'Show only ' + (c.name || c.id) + ' recordings', group: 'Cameras',
      icon: 'filter', scope: c.id,
      run: function () { router.go('/recordings', { cameras: c.id }); }
    });
  });
  (store.get('savedViews') || []).forEach(function (v) {
    items.push({
      name: v.name, group: 'Saved views', icon: 'bookmark',
      scope: v.count === undefined ? '' : String(v.count),
      run: function () { router.go('/recordings', v.query || {}); }
    });
  });
  items.push({
    name: 'Collapse the filter rail', group: 'Appearance', icon: 'layers',
    run: function () { toggleRail(); }
  });
  return items;
}

function saveClipFrom(cam) {
  var t = toast.progress('Saving the last 30 s from ' + (cam.name || cam.id) + '…');
  api.saveClip(cam.id).then(function (res) {
    t.close();
    toast.success('Clip saved from ' + (cam.name || cam.id), {
      detail: (res && res.filename) || '',
      action: { label: 'View', variant: 'secondary', onClick: function () {
        router.go('/recordings', {});
      } }
    });
  }, function (err) {
    t.close();
    toast.error('Could not save the clip from ' + (cam.name || cam.id) + '.', {
      detail: api.describe(err),
      retry: function () { saveClipFrom(cam); }
    });
  });
}

function openPalette() {
  if (window.innerWidth < 1024) return;      /* desktop only, by design */
  palette({ items: paletteItems() });
}

/* ==========================================================================
   POLLING — paused whenever the document is hidden
   ========================================================================= */
function createPoller(fn, intervalMs) {
  var timer = null;
  var stopped = false;
  var running = false;
  function tick() {
    if (stopped) { running = false; return; }
    timer = null;
    var p = fn();
    var next = function () {
      if (stopped || !running) return;
      timer = window.setTimeout(tick, intervalMs);
    };
    if (p && p.then) p.then(next, next);
    else next();
  }
  return {
    start: function () {
      if (running || stopped) return;
      running = true;
      tick();
    },
    stop: function () {
      running = false;
      if (timer) { clearTimeout(timer); timer = null; }
    },
    dispose: function () { stopped = true; this.stop(); }
  };
}

/* ==========================================================================
   BOOT
   ========================================================================= */
function boot() {
  installSprite();
  store.set({
    theme: readTheme(),
    savedViews: loadSavedViews(),
    railCollapsed: readLocal('at:rail-collapsed', false) === true,
    density: readLocal('at:density', 'comfortable')
  });

  var root = document.getElementById('app');
  var shell = buildShell(root);
  window.__atShell = shell;   /* debugging handle only; nothing reads it */

  /* --- the content gutter ------------------------------------------------ */
  var wide = window.matchMedia('(min-width: 640px)');
  function syncPad() {
    root.style.setProperty('--page-pad', wide.matches ? 'var(--s-6)' : 'var(--s-4)');
  }
  syncPad();
  if (wide.addEventListener) wide.addEventListener('change', syncPad);
  else if (wide.addListener) wide.addListener(syncPad);
  shell.main.style.paddingLeft = 'calc(var(--inset-left) + var(--page-pad))';
  shell.main.style.paddingRight = 'calc(var(--inset-right) + var(--page-pad))';

  /* --- rail collapse ----------------------------------------------------- */
  window.__atToggleRail = toggleRail;

  /* --- chrome rendering -------------------------------------------------- */
  function renderChrome() {
    var c = store.get('chrome') || {};

    shell.title.firstChild.nodeValue = c.title || 'Animal Tracker';
    shell.subtitle.textContent = c.subtitle || '';
    shell.subtitle.hidden = !c.subtitle;
    document.title = (c.title ? c.title + ' · ' : '') + 'Animal Tracker';

    /* Re-appending the same nodes would drop focus mid-interaction, and the
       view calls setChrome() every time a count changes. Only rebuild when
       the set of nodes actually differs. */
    var acts = c.actions || [];
    var same = shell._actions && shell._actions.length === acts.length &&
      shell._actions.every(function (el, i) { return el === acts[i]; });
    if (!same) {
      clear(shell.actions);
      acts.forEach(function (el) { shell.actions.appendChild(el); });
      shell._actions = acts.slice();
    }

    if (c.toolbar) {
      if (shell.toolbarSlot.firstChild !== c.toolbar) {
        clear(shell.toolbarSlot).appendChild(c.toolbar);
      }
      shell.toolbarSlot.hidden = false;
    } else {
      clear(shell.toolbarSlot);
      shell.toolbarSlot.hidden = true;
    }

    if (c.rail) {
      if (shell.railViewSlot.firstChild !== c.rail) {
        clear(shell.railViewSlot).appendChild(c.rail);
      }
      shell.railViewSlot.hidden = false;
    } else {
      clear(shell.railViewSlot);
      shell.railViewSlot.hidden = true;
    }

    /* The selection bar is chrome, not a modal: a child of .shell. */
    var existing = root.querySelector(':scope > .selbar');
    if (c.selbar) {
      if (existing !== c.selbar) {
        if (existing) existing.remove();
        root.appendChild(c.selbar);
      }
    } else if (existing) {
      existing.remove();
    }

    var mods = ['shell'];
    if (c.norail) mods.push('shell--norail');
    if (store.get('railCollapsed')) mods.push('shell--rail-collapsed');
    if (c.toolbar) mods.push('shell--toolbar');
    if (c.selbar) mods.push('shell--selbar');
    (c.mods || []).forEach(function (m) { mods.push('shell--' + m); });
    root.className = mods.join(' ');
    shell.rail.hidden = !!c.norail;
  }
  store.select(['chrome', 'railCollapsed'], renderChrome);

  store.select(['cameras', 'camerasError'], function () { renderRailCameras(shell); });
  store.select(['system'], function () { renderRailFoot(shell); });
  store.select(['savedViews'], function () { renderSavedViews(shell); });

  renderRailCameras(shell);
  renderRailFoot(shell);
  renderSavedViews(shell);

  /* --- connection banner ------------------------------------------------- */
  var offlineToast = null;
  store.select(['connected'], function (s) {
    if (!s.connected && !offlineToast) {
      offlineToast = toast.error('Disconnected from the Animal Tracker server.', {
        detail: 'Cached views stay readable; changes are disabled until it answers again.'
      });
    } else if (s.connected && offlineToast) {
      offlineToast.close();
      offlineToast = null;
    }
  });

  /* --- routes ------------------------------------------------------------ */
  router.register('/', recordingsView);
  router.register('/recordings', recordingsView);
  router.register('/live', placeholderView('Live', 'live'));
  router.register('/monitor', placeholderView('Monitor', 'monitor'));
  router.register('/settings', placeholderView('Settings', 'settings'));
  router.register('/clips/:path*', placeholderView('Clip detail', 'film'));
  router.setFallback(notFoundView());

  router.subscribe(function (ctx) {
    var path = ctx.path;
    function markTabs(list) {
      list.forEach(function (el) {
        var p = el.dataset.path;
        var active = path === p || path.indexOf(p + '/') === 0 ||
          (p === '/recordings' && path === '/');
        if (active) el.setAttribute('aria-current', 'page');
        else el.removeAttribute('aria-current');
      });
    }
    markTabs(shell.tabs);
    markTabs(shell.tabbarItems);

    /* Keep the global field in step with the URL without stealing focus. */
    if (document.activeElement !== shell.searchInput) {
      shell.searchInput.value = ctx.query.q || '';
      shell.searchField.classList.toggle('search--filled', !!shell.searchInput.value);
    }

    renderSavedViews(shell);

    /* Focus lands on the view's heading; the shell announces the surface. */
    window.setTimeout(function () {
      var heading = shell.main.querySelector('h1');
      var target = heading || shell.main;
      if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
      if (!isOverlayOpen()) {
        try { target.focus({ preventScroll: true }); } catch (e) { target.focus(); }
      }
      shell.live.textContent = (store.get('chrome') || {}).title || '';
    }, 0);
  });

  router.interceptLinks(document.body);

  if (window.location.pathname === '/app' || window.location.pathname === '/app/') {
    window.history.replaceState(null, '', router.href('/recordings', router.current
      ? router.current.query : {}));
  }
  router.start(shell.main);

  /* --- polling ----------------------------------------------------------- */
  var camPoll = createPoller(function () {
    return api.cameras().then(function (res) {
      store.set({
        cameras: res.cameras || [],
        timezone: res.timezone || '',
        camerasError: null
      });
    }, function (err) {
      if (api.isAbort(err)) return;
      store.set({ camerasError: err });
    });
  }, CAMERA_POLL_MS);

  var sysPoll = createPoller(function () {
    if (window.innerWidth < 1024) return null;   /* the footer is rail-only */
    return api.monitor().then(function (res) {
      store.set({ system: res.system || null });
    }, function () { /* the rail footer degrades to "loading"; not fatal */ });
  }, MONITOR_POLL_MS);

  function syncVisibility() {
    var visible = document.visibilityState !== 'hidden';
    store.set({ visible: visible });
    if (visible) { camPoll.start(); sysPoll.start(); }
    else { camPoll.stop(); sysPoll.stop(); }
  }
  on(document, 'visibilitychange', syncVisibility);
  syncVisibility();

  /* A pending undo deadline must not be lost when the tab goes away. */
  on(window, 'pagehide', function () { toast.flush(); });

  refreshSavedCounts();

  /* --- global keyboard --------------------------------------------------- */
  installKeyboard(shell);
}

function toggleRail() {
  var next = !store.get('railCollapsed');
  store.set({ railCollapsed: next });
  writeLocal('at:rail-collapsed', next);
}

function isTypingTarget(el) {
  if (!el) return false;
  var tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

function installKeyboard(shell) {
  var chord = null;
  var chordTimer = null;

  on(document, 'keydown', function (ev) {
    if (ev.defaultPrevented) return;

    /* Cmd/Ctrl+K works even inside a field — it is the escape hatch. */
    if ((ev.metaKey || ev.ctrlKey) && (ev.key === 'k' || ev.key === 'K')) {
      ev.preventDefault();
      openPalette();
      return;
    }
    if (ev.key === 'Escape') {
      if (closeTop()) ev.preventDefault();
      return;
    }
    if (isTypingTarget(ev.target)) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;

    if (chord === 'g') {
      var map = { l: '/live', r: '/recordings', m: '/monitor', s: '/settings' };
      var dest = map[ev.key.toLowerCase()];
      chord = null;
      if (dest) { ev.preventDefault(); router.go(dest, {}); }
      return;
    }

    if (ev.key === '/') {
      ev.preventDefault();
      if (window.innerWidth >= 1024) shell.searchInput.focus();
      else {
        var mobileSearch = document.querySelector('.rail .search__input, .sheet .search__input');
        if (mobileSearch) mobileSearch.focus();
        else shell.searchInput.focus();
      }
      return;
    }
    if (ev.key === 'g') {
      chord = 'g';
      if (chordTimer) clearTimeout(chordTimer);
      chordTimer = window.setTimeout(function () { chord = null; }, 1200);
      return;
    }
    if (ev.key === '?') {
      ev.preventDefault();
      openShortcuts();
    }
  });
}

function openShortcuts() {
  var rows = [
    ['/', 'Focus search'],
    [isMac() ? '⌘K' : 'Ctrl K', 'Command palette (desktop)'],
    ['g then l / r / m / s', 'Jump to Live, Recordings, Monitor, Settings'],
    ['Arrows', 'Move through the clip grid or the calendar'],
    ['Enter', 'Open the focused clip or day'],
    ['Space', 'Toggle selection in selection mode'],
    ['Shift + click', 'Select a range'],
    [isMac() ? '⌘A' : 'Ctrl A', 'Select everything on screen'],
    ['Escape', 'Leave selection mode, or close the top panel'],
    ['?', 'This list']
  ];
  var body = h('div.stack.stack--tight');
  rows.forEach(function (r) {
    body.appendChild(h('div.row',
      h('kbd.kbd', { text: r[0] }),
      h('span.t-sm.t-2', { text: r[1] })));
  });
  dialog({
    role: 'dialog', title: 'Keyboard shortcuts', width: 520, content: body,
    actions: [{ label: 'Close', variant: 'secondary', value: null, focus: true }]
  });
}

/* --------------------------------------------------------------------------
   PLACEHOLDER VIEWS
   Live, Monitor, Settings and Clip detail are written against this core in
   the phases that follow. Until then the tab lands somewhere honest rather
   than on a blank page.
   ------------------------------------------------------------------------ */
function placeholderView(title, iconName) {
  return {
    mount: function (root, ctx) {
      store.setChrome({
        title: title, subtitle: '', actions: [], toolbar: null,
        rail: null, norail: title === 'Live' || title === 'Settings',
        selbar: null, mods: title === 'Live' ? ['no-blur'] : []
      });
      root.appendChild(h('h1.visually-hidden', { tabIndex: -1, text: title }));
      root.appendChild(h('div.empty',
        h('div.empty__art', icon(iconName, { size: 'lg' })),
        h('h2.empty__title', { text: title + ' is not built yet' }),
        h('p.empty__body',
          'This surface arrives in a later phase of the rewrite. The old page is ' +
          'still served and still works.'),
        h('div.empty__actions',
          h('a.btn.btn--primary', { href: ctx.path },
            h('span.btn__label', 'Open the old ' + title + ' page')),
          h('a.btn.btn--secondary', { href: router.href('/recordings') },
            h('span.btn__label', 'Back to Recordings')))));
    },
    unmount: function () {}
  };
}

function notFoundView() {
  return {
    mount: function (root, ctx) {
      store.setChrome({ title: 'Not found', subtitle: '', actions: [], toolbar: null,
        rail: null, norail: true, selbar: null, mods: [] });
      root.appendChild(h('h1.visually-hidden', { tabIndex: -1, text: 'Not found' }));
      root.appendChild(h('div.empty.empty--error',
        h('div.empty__art', icon('alert', { size: 'lg' })),
        h('h2.empty__title', 'No screen answers to that address'),
        h('p.empty__body', 'The app has four surfaces: Live, Recordings, Monitor and Settings.'),
        h('p.empty__endpoint', { text: ctx.url }),
        h('div.empty__actions',
          h('a.btn.btn--primary', { href: router.href('/recordings') },
            h('span.btn__label', 'Go to Recordings')))));
    },
    unmount: function () {}
  };
}

/* Modules are deferred, so the DOM is already parsed; the guard is for the
   case where this file is ever loaded with `async` from the head. */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
