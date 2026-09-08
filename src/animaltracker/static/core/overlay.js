/* ============================================================================
   core/overlay.js — every surface that sits above the page: dialog, bottom
   sheet, popover menu, command palette, and the focus machinery they share.

   THE RULES, ENFORCED HERE SO NO CALLER HAS TO REMEMBER THEM
     · document.activeElement is stored on open and restored on close.
     · Focus moves INTO the surface (the least destructive control in a
       destructive dialog; the first field otherwise).
     · Tab is trapped. Escape closes. Outside click closes (menus and
       non-alert dialogs).
     · Everything outside the surface gets `inert` + aria-hidden, and the body
       scroll is locked with position:fixed/top:-Npx and restored exactly.
     · Overlays are appended to <body> as siblings of .shell, so no
       transformed ancestor can trap them in a stacking context.
     · The exit animation runs on .is-closing for one duration, then the node
       is removed.

   PUBLIC API
     focusTrap(el, opts)          -> { release() }
     lockScroll() / unlockScroll()
     dialog(opts)                 -> handle { el, body, close(v), result }
     sheet(opts)                  -> handle { el, body, close(v), result }
     menu(anchor, items, opts)    -> handle { el, close() }
     palette(opts)                -> handle { el, close() }
     isOverlayOpen()              -> boolean

   dialog(opts)
     { title, body, stakes, tone:'danger'|'plain', icon:'alert',
       content: Node | function(host),         // custom body
       width: 720,                             // px cap, default 440 via CSS
       actions: [{ label, value, variant, focus:true, disabled }],
       role: 'alertdialog' | 'dialog',
       dismissible: true,                      // Escape / outside click
       onClose: function (value) {} }
     -> handle.result is a Promise of the chosen action's `value` (null when
        dismissed).

   sheet(opts)
     { title, items:[{label, icon, danger, disabled, value, onSelect}],
       content, footer, snap:'peek'|'half'|'full', cancelLabel,
       mobileOnly:true, onClose }
     Items render in DOM order and the destructive one belongs ABOVE Cancel.
   ========================================================================= */

import { h, clear, on } from './dom.js';
import { icon } from './icons.js';

var FOCUSABLE = 'a[href],area[href],button:not([disabled]),input:not([disabled])' +
  ':not([type="hidden"]),select:not([disabled]),textarea:not([disabled]),' +
  'iframe,object,embed,summary,video[controls],audio[controls],[tabindex]';

var openCount = 0;
var scrollY = 0;
var stack = [];

function focusables(root) {
  var out = [];
  var nodes = root.querySelectorAll(FOCUSABLE);
  for (var i = 0; i < nodes.length; i++) {
    var el = nodes[i];
    if (el.disabled) continue;
    if (el.getAttribute('tabindex') === '-1') continue;
    if (el.offsetParent === null && el.getClientRects().length === 0) continue;
    out.push(el);
  }
  return out;
}

export function isOverlayOpen() { return stack.length > 0; }

export function lockScroll() {
  openCount += 1;
  if (openCount > 1) return;
  scrollY = window.pageYOffset || document.documentElement.scrollTop || 0;
  document.body.style.top = (-scrollY) + 'px';
  document.body.classList.add('is-locked');
}

export function unlockScroll() {
  openCount = Math.max(0, openCount - 1);
  if (openCount > 0) return;
  document.body.classList.remove('is-locked');
  document.body.style.top = '';
  window.scrollTo(0, scrollY);
}

/* The shell is the only thing outside an overlay; making it inert is both the
   ARIA story and the pointer story. Nested overlays share one count. */
function setOutsideInert(flag) {
  var shell = document.getElementById('app');
  if (!shell) return;
  if (flag) {
    shell.setAttribute('inert', '');
    shell.setAttribute('aria-hidden', 'true');
  } else {
    shell.removeAttribute('inert');
    shell.removeAttribute('aria-hidden');
  }
}

/**
 * Trap Tab inside `el` and restore focus on release.
 * opts: { onEscape, initial: Element|selector, returnTo: Element }
 */
export function focusTrap(el, opts) {
  var o = opts || {};
  var previous = o.returnTo || document.activeElement;

  function onKeyDown(ev) {
    if (ev.key === 'Escape' || ev.key === 'Esc') {
      if (o.onEscape) { ev.stopPropagation(); ev.preventDefault(); o.onEscape(); }
      return;
    }
    if (ev.key !== 'Tab') return;
    var list = focusables(el);
    if (!list.length) { ev.preventDefault(); el.focus(); return; }
    var first = list[0];
    var last = list[list.length - 1];
    var active = document.activeElement;
    if (ev.shiftKey && (active === first || !el.contains(active))) {
      ev.preventDefault(); last.focus();
    } else if (!ev.shiftKey && (active === last || !el.contains(active))) {
      ev.preventDefault(); first.focus();
    }
  }

  var offKey = on(document, 'keydown', onKeyDown, true);

  var target = null;
  if (o.initial) {
    target = typeof o.initial === 'string' ? el.querySelector(o.initial) : o.initial;
  }
  if (!target) target = focusables(el)[0] || el;
  /* rAF: an element inside an entering animation can measure as unfocusable. */
  window.requestAnimationFrame(function () {
    try { target.focus({ preventScroll: true }); } catch (e) { try { target.focus(); } catch (e2) {} }
  });

  return {
    release: function () {
      offKey();
      if (previous && previous.isConnected && typeof previous.focus === 'function') {
        try { previous.focus({ preventScroll: true }); } catch (e) { previous.focus(); }
      }
    }
  };
}

/* --- shared open/close plumbing ------------------------------------------ */

function openSurface(spec) {
  /* spec: { host, surface, scrim, dismissible, onDismiss, initial, modal } */
  var entry = { host: spec.host, closing: false };
  stack.push(entry);
  document.body.appendChild(spec.host);
  if (spec.modal !== false) {
    lockScroll();
    setOutsideInert(true);
  }
  entry.modal = spec.modal !== false;

  entry.trap = focusTrap(spec.surface, {
    onEscape: spec.dismissible === false ? null : spec.onDismiss,
    initial: spec.initial
  });

  entry.close = function (after) {
    if (entry.closing) return;
    entry.closing = true;
    var i = stack.indexOf(entry);
    if (i >= 0) stack.splice(i, 1);
    spec.surface.classList.add('is-closing');
    if (spec.scrim) spec.scrim.classList.add('is-closing');
    entry.trap.release();
    if (entry.modal) {
      setOutsideInert(stack.length > 0);
      unlockScroll();
    }
    window.setTimeout(function () {
      if (spec.host.parentNode) spec.host.parentNode.removeChild(spec.host);
      if (spec.scrim && spec.scrim.parentNode) spec.scrim.parentNode.removeChild(spec.scrim);
      if (after) after();
    }, spec.exit === undefined ? 220 : spec.exit);
  };

  return entry;
}

function makeScrim(onClick) {
  var scrim = h('div.scrim');
  if (onClick) scrim.addEventListener('click', onClick);
  document.body.appendChild(scrim);
  return scrim;
}

/* --- DIALOG -------------------------------------------------------------- */

export function dialog(opts) {
  var o = opts || {};
  var resolveResult;
  var result = new Promise(function (res) { resolveResult = res; });

  var host = h('div.dialog-host');
  var el = h('div.dialog', {
    role: o.role || 'alertdialog',
    'aria-modal': 'true'
  });
  if (o.width) el.style.width = 'min(' + o.width + 'px, 100%)';

  var titleId = 'at-dlg-title-' + Math.random().toString(36).slice(2, 8);
  var bodyId = titleId + '-body';

  if (o.tone === 'danger' || o.icon) {
    el.appendChild(h('div.dialog__icon', icon(o.icon || 'alert', { size: 'lg' })));
  }
  if (o.title) {
    el.appendChild(h('h2.dialog__title#' + titleId, { text: o.title }));
    el.setAttribute('aria-labelledby', titleId);
  }
  if (o.body) {
    el.appendChild(h('p.dialog__body#' + bodyId, { text: o.body }));
    el.setAttribute('aria-describedby', bodyId);
  }
  if (o.stakes) el.appendChild(h('span.dialog__stakes', { text: o.stakes }));

  var content = null;
  if (o.content) {
    content = h('div.stack');
    if (typeof o.content === 'function') o.content(content);
    else content.appendChild(o.content);
    el.appendChild(content);
  }

  var handle = null;
  function finish(value) {
    if (!handle) return;
    handle.closed = true;
    entry.close(function () {
      if (o.onClose) o.onClose(value);
      resolveResult(value);
    });
  }

  var initial = null;
  if (o.actions && o.actions.length) {
    var row = h('div.dialog__actions');
    for (var i = 0; i < o.actions.length; i++) {
      (function (action) {
        var btn = h('button.btn', {
          type: 'button',
          'class': 'btn--' + (action.variant || 'secondary'),
          disabled: !!action.disabled
        }, h('span.btn__label', { text: action.label }));
        btn.addEventListener('click', function () {
          if (action.onSelect) action.onSelect();
          if (action.keepOpen) return;
          finish(action.value === undefined ? action.label : action.value);
        });
        if (action.focus) initial = btn;
        row.appendChild(btn);
      }(o.actions[i]));
    }
    el.appendChild(row);
  }

  host.appendChild(el);
  var dismissible = o.dismissible !== false;
  var scrim = makeScrim(dismissible ? function () { finish(null); } : null);

  var entry = openSurface({
    host: host, surface: el, scrim: scrim,
    dismissible: dismissible,
    onDismiss: function () { finish(null); },
    initial: initial || (o.initialFocus || null)
  });

  handle = {
    el: el,
    body: content,
    closed: false,
    result: result,
    close: function (value) { finish(value === undefined ? null : value); }
  };
  return handle;
}

/* --- BOTTOM SHEET -------------------------------------------------------- */

export function sheet(opts) {
  var o = opts || {};
  var resolveResult;
  var result = new Promise(function (res) { resolveResult = res; });

  var host = h('div.sheet-host');
  if (o.mobileOnly !== false) host.classList.add('sheet-host--mobile-only');

  var el = h('div.sheet', {
    role: 'dialog',
    'aria-modal': 'true',
    'class': o.snap ? 'sheet--' + o.snap : 'sheet--half'
  });

  var titleId = 'at-sheet-title-' + Math.random().toString(36).slice(2, 8);
  var handleBtn = h('button.sheet__handle', {
    type: 'button',
    'aria-label': 'Close ' + (o.title || 'panel')
  });
  el.appendChild(handleBtn);

  if (o.title) {
    el.appendChild(h('div.sheet__head',
      h('h2.sheet__title#' + titleId, { text: o.title }),
      o.headActions || null));
    el.setAttribute('aria-labelledby', titleId);
  }

  var body = h('div.sheet__body');
  el.appendChild(body);

  function finish(value) {
    entry.close(function () {
      if (o.onClose) o.onClose(value);
      resolveResult(value === undefined ? null : value);
    });
  }

  if (o.items && o.items.length) {
    var list = h('ul.sheet__list', { role: 'list' });
    for (var i = 0; i < o.items.length; i++) {
      (function (item) {
        var btn = h('button.sheet__item', {
          type: 'button',
          'class': item.danger ? 'sheet__item--danger' : null,
          'aria-disabled': item.disabled ? 'true' : null
        });
        if (item.icon) btn.appendChild(icon(item.icon));
        btn.appendChild(h('span', { text: item.label }));
        btn.addEventListener('click', function () {
          if (item.disabled) return;
          if (item.onSelect) item.onSelect();
          finish(item.value === undefined ? item.label : item.value);
        });
        list.appendChild(h('li', btn));
      }(o.items[i]));
    }
    body.appendChild(list);
    var cancel = h('button.sheet__cancel', { type: 'button' },
      o.cancelLabel || 'Cancel');
    cancel.addEventListener('click', function () { finish(null); });
    body.appendChild(cancel);
  }

  if (o.content) {
    if (typeof o.content === 'function') o.content(body);
    else body.appendChild(o.content);
  }
  if (o.footer) el.appendChild(o.footer);

  handleBtn.addEventListener('click', function () { finish(null); });

  host.appendChild(el);
  var scrim = makeScrim(function () { finish(null); });

  var entry = openSurface({
    host: host, surface: el, scrim: scrim,
    dismissible: true,
    onDismiss: function () { finish(null); },
    initial: o.initialFocus || null,
    exit: 160
  });

  /* Drag-to-dismiss on the grab handle. touch-action:none is already on the
     handle in CSS; will-change is added only for the life of the gesture. */
  var dragFrom = 0;
  var dragging = false;
  handleBtn.addEventListener('pointerdown', function (ev) {
    dragFrom = ev.clientY;
    dragging = true;
    el.classList.add('is-dragging');
    try { handleBtn.setPointerCapture(ev.pointerId); } catch (e) {}
  });
  handleBtn.addEventListener('pointermove', function (ev) {
    if (!dragging) return;
    var dy = Math.max(0, ev.clientY - dragFrom);
    el.style.setProperty('--sheet-drag', dy + 'px');
  });
  function endDrag(ev) {
    if (!dragging) return;
    dragging = false;
    el.classList.remove('is-dragging');
    var dy = Math.max(0, (ev.clientY || 0) - dragFrom);
    el.style.setProperty('--sheet-drag', '0px');
    if (dy > 110) finish(null);
  }
  handleBtn.addEventListener('pointerup', endDrag);
  handleBtn.addEventListener('pointercancel', endDrag);

  return {
    el: el,
    body: body,
    result: result,
    close: function (value) { finish(value); }
  };
}

/* --- POPOVER MENU -------------------------------------------------------- */

/**
 * menu(anchorEl, items, opts)
 *   items: [{ label, icon, hint, checked, current, danger, disabled,
 *             onSelect, separator, group }]
 * Roving tabindex, arrow keys, Escape, outside click, and it closes on scroll.
 */
export function menu(anchor, items, opts) {
  var o = opts || {};
  var el = h('div.menu', { role: 'menu' });
  if (o.label) el.setAttribute('aria-label', o.label);

  var buttons = [];
  for (var i = 0; i < items.length; i++) {
    var item = items[i];
    if (item.separator) { el.appendChild(h('hr.menu__sep')); continue; }
    if (item.group) { el.appendChild(h('div.menu__label', { text: item.group })); continue; }
    (function (it) {
      var btn = h('button.menu__item', {
        type: 'button',
        role: it.checked === undefined ? 'menuitem' : 'menuitemradio',
        tabIndex: -1,
        'class': it.danger ? 'menu__item--danger' : null,
        'aria-checked': it.checked === undefined ? null : (it.checked ? 'true' : 'false'),
        'aria-current': it.current ? 'true' : null,
        'aria-disabled': it.disabled ? 'true' : null
      });
      if (it.icon) btn.appendChild(icon(it.icon, { size: 'sm' }));
      btn.appendChild(h('span', { text: it.label }));
      if (it.hint) btn.appendChild(h('span.menu__hint', { text: it.hint }));
      btn.addEventListener('click', function () {
        if (it.disabled) return;
        close();
        if (it.onSelect) it.onSelect();
      });
      buttons.push(btn);
      el.appendChild(btn);
    }(item));
  }

  var closed = false;
  var offs = [];
  function close() {
    if (closed) return;
    closed = true;
    for (var k = 0; k < offs.length; k++) offs[k]();
    var i2 = stack.indexOf(entry);
    if (i2 >= 0) stack.splice(i2, 1);
    el.classList.add('is-closing');
    trap.release();
    window.setTimeout(function () {
      if (el.parentNode) el.parentNode.removeChild(el);
      if (o.onClose) o.onClose();
    }, 120);
  }

  document.body.appendChild(el);

  /* Position with a viewport flip. No positioning library, no observers. */
  var rect = anchor.getBoundingClientRect();
  var box = el.getBoundingClientRect();
  var gap = 6;
  var x = rect.left;
  var y = rect.bottom + gap;
  if (o.align === 'right' || x + box.width > window.innerWidth - 8) {
    x = Math.max(8, rect.right - box.width);
    el.classList.add('menu--left');
  }
  if (y + box.height > window.innerHeight - 8) {
    y = Math.max(8, rect.top - box.height - gap);
    el.classList.add(el.classList.contains('menu--left') ? 'menu--upleft' : 'menu--up');
  }
  el.style.setProperty('--menu-x', Math.round(x) + 'px');
  el.style.setProperty('--menu-y', Math.round(y) + 'px');

  var entry = { close: close, host: el, modal: false };
  stack.push(entry);

  var trap = focusTrap(el, { onEscape: close, initial: buttons[0] });

  /* Roving tabindex inside the menu. */
  function focusAt(index) {
    if (!buttons.length) return;
    var n = ((index % buttons.length) + buttons.length) % buttons.length;
    buttons[n].focus();
  }
  offs.push(on(el, 'keydown', function (ev) {
    var idx = buttons.indexOf(document.activeElement);
    if (ev.key === 'ArrowDown') { ev.preventDefault(); focusAt(idx + 1); }
    else if (ev.key === 'ArrowUp') { ev.preventDefault(); focusAt(idx - 1); }
    else if (ev.key === 'Home') { ev.preventDefault(); focusAt(0); }
    else if (ev.key === 'End') { ev.preventDefault(); focusAt(buttons.length - 1); }
  }));

  /* Outside click / scroll close. Deferred one tick so the click that opened
     the menu does not immediately close it. */
  window.setTimeout(function () {
    if (closed) return;
    offs.push(on(document, 'pointerdown', function (ev) {
      if (!el.contains(ev.target) && ev.target !== anchor) close();
    }, true));
    offs.push(on(window, 'scroll', close, true));
    offs.push(on(window, 'resize', close));
  }, 0);

  return { el: el, close: close };
}

/* --- COMMAND PALETTE ----------------------------------------------------- */

/**
 * palette({ items, placeholder, onRun })
 *   items: [{ id, name, group, scope, icon, run }]
 * A combobox: focus stays in the input and aria-activedescendant moves, so
 * typing never loses the caret. Desktop only — app.js gates it at 1024px.
 */
export function palette(opts) {
  var o = opts || {};
  var all = o.items || [];

  var host = h('div.palette-host');
  var el = h('div.palette', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Command palette' });

  var listId = 'at-palette-list';
  var input = h('input', {
    type: 'text',
    role: 'combobox',
    autocomplete: 'off',
    autocorrect: 'off',
    spellcheck: 'false',
    'aria-expanded': 'true',
    'aria-controls': listId,
    'aria-autocomplete': 'list',
    placeholder: o.placeholder || 'Search actions and views…'
  });
  el.appendChild(h('div.palette__input', icon('search'), input));

  var list = h('div.palette__list#' + listId, { role: 'listbox', 'aria-label': 'Commands' });
  el.appendChild(list);
  el.appendChild(h('div.palette__foot',
    h('span', '↑↓ to navigate'),
    h('span', 'Enter to run'),
    h('span', 'Esc to close')));

  var rows = [];
  var active = 0;

  function score(item, needle) {
    var hay = (item.name + ' ' + (item.group || '') + ' ' + (item.scope || '')).toLowerCase();
    if (!needle) return 1;
    var idx = hay.indexOf(needle);
    if (idx >= 0) return 100 - idx;
    /* subsequence match, so "tgd" finds "Toggle grid density" */
    var j = 0;
    for (var i = 0; i < hay.length && j < needle.length; i++) {
      if (hay.charAt(i) === needle.charAt(j)) j++;
    }
    return j === needle.length ? 10 : 0;
  }

  function render() {
    var needle = input.value.trim().toLowerCase();
    var matched = [];
    for (var i = 0; i < all.length; i++) {
      var s = score(all[i], needle);
      if (s > 0) matched.push({ item: all[i], s: s, i: i });
    }
    matched.sort(function (a, b) { return b.s - a.s || a.i - b.i; });

    clear(list);
    rows = [];
    if (!matched.length) {
      list.appendChild(h('div.palette__empty', { text: 'No command matches “' + input.value + '”.' }));
      input.removeAttribute('aria-activedescendant');
      return;
    }
    var lastGroup = null;
    for (var m = 0; m < matched.length; m++) {
      var item = matched[m].item;
      if (item.group && item.group !== lastGroup) {
        lastGroup = item.group;
        list.appendChild(h('div.palette__group', { text: item.group }));
      }
      (function (it, index) {
        var row = h('div.palette__item', {
          role: 'option',
          id: 'at-pal-' + index,
          'aria-selected': 'false'
        });
        if (it.icon) row.appendChild(icon(it.icon, { size: 'sm' }));
        row.appendChild(h('span.palette__name', { text: it.name }));
        if (it.scope) row.appendChild(h('span.palette__scope', { text: it.scope }));
        row.addEventListener('click', function () { run(it, row); });
        row.addEventListener('mousemove', function () { setActive(index); });
        rows.push({ el: row, item: it });
        list.appendChild(row);
      }(item, m));
    }
    setActive(0);
  }

  function setActive(index) {
    if (!rows.length) return;
    active = Math.max(0, Math.min(rows.length - 1, index));
    for (var i = 0; i < rows.length; i++) {
      rows[i].el.setAttribute('aria-selected', i === active ? 'true' : 'false');
    }
    var el2 = rows[active].el;
    input.setAttribute('aria-activedescendant', el2.id);
    if (el2.scrollIntoView) el2.scrollIntoView({ block: 'nearest' });
  }

  function run(item, row) {
    if (row) row.setAttribute('aria-busy', 'true');
    close();
    window.setTimeout(function () {
      try { item.run(); } catch (err) { if (window.console) console.error('[palette]', err); }
    }, 0);
  }

  input.addEventListener('input', render);
  input.addEventListener('keydown', function (ev) {
    if (ev.key === 'ArrowDown') { ev.preventDefault(); setActive(active + 1); }
    else if (ev.key === 'ArrowUp') { ev.preventDefault(); setActive(active - 1); }
    else if (ev.key === 'Home' && !input.value) { ev.preventDefault(); setActive(0); }
    else if (ev.key === 'End' && !input.value) { ev.preventDefault(); setActive(rows.length - 1); }
    else if (ev.key === 'Enter') {
      ev.preventDefault();
      if (rows[active]) run(rows[active].item, rows[active].el);
    }
  });

  host.appendChild(el);
  var scrim = makeScrim(function () { close(); });
  var entry = openSurface({
    host: host, surface: el, scrim: scrim,
    dismissible: true,
    onDismiss: function () { close(); },
    initial: input
  });

  function close() { entry.close(o.onClose); }

  render();
  return { el: el, close: close };
}

/** Close the topmost overlay, if any. Used by the global Escape handler. */
export function closeTop() {
  if (!stack.length) return false;
  var top = stack[stack.length - 1];
  if (top.close) top.close();
  return true;
}
