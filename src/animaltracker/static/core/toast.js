/* ============================================================================
   core/toast.js — the only way the app reports a mutation, a failure or an
   undoable action. Replaces every alert() and every confirm() on a reversible
   path.

   The host is a fixed strip measuring from --chrome-bottom, so it clears the
   phone tab bar and the desktop home indicator without a magic number. It is
   role="status" aria-live="polite"; a persistent error is role="alert".

   THE UNDO CONTRACT
   toast('3 recordings deleted', { kind:'danger', undo:{...} }) shows a visible
   6-second drain bar (transform: scaleX, never width). Hovering or focusing
   the toast pauses BOTH the CSS animation and this module's timer, so the
   deadline the user sees is the deadline that fires.

     var t = toast('1 recording deleted', {
       kind: 'danger',
       detail: '0.08 MB freed',
       undo: {
         label: 'Undo',
         onUndo:   function () { ...put it back... },
         onExpire: function () { ...actually delete... }   // fires once
       }
     });
     t.close();                 // dismiss without firing onExpire
     t.update({ title, detail });

   PUBLIC API
     toast(message, opts) -> handle { close(fireExpire?), update(patch), el }
     toast.success / .info / .danger / .error / .progress  (kind shorthands)
     toast.clear()
   ========================================================================= */

import { h, clear } from './dom.js';
import { icon } from './icons.js';

var UNDO_MS = 6000;      /* must equal --dur-undo */
var TIMEOUTS = { success: 4000, info: 5000, danger: UNDO_MS, progress: 0, error: 0 };
var MAX_VISIBLE = 3;

var host = null;
var live = [];

function ensureHost() {
  if (host && host.isConnected) return host;
  host = document.getElementById('at-toast-host');
  if (!host) {
    host = h('div.toast-host#at-toast-host', {
      role: 'status',
      'aria-live': 'polite',
      'aria-atomic': 'false'
    });
    document.body.appendChild(host);
  }
  return host;
}

var ICONS = {
  success: 'check',
  info: 'info',
  danger: 'trash',
  error: 'alert',
  progress: 'refresh'
};

function dismiss(entry, fireExpire) {
  if (entry.done) return;
  entry.done = true;
  if (entry.timer) { clearTimeout(entry.timer); entry.timer = null; }
  var i = live.indexOf(entry);
  if (i >= 0) live.splice(i, 1);

  if (fireExpire && entry.onExpire) {
    var fn = entry.onExpire;
    entry.onExpire = null;
    try { fn(); } catch (err) { if (window.console) console.error('[toast] onExpire failed', err); }
  }

  var el = entry.el;
  el.classList.add('is-closing');
  window.setTimeout(function () {
    if (el.parentNode) el.parentNode.removeChild(el);
  }, 200);
}

function startTimer(entry, ms) {
  if (!ms) return;
  entry.remaining = ms;
  entry.startedAt = Date.now();
  entry.timer = window.setTimeout(function () { dismiss(entry, true); }, ms);
}

function pause(entry) {
  if (!entry.timer || entry.paused) return;
  entry.paused = true;
  clearTimeout(entry.timer);
  entry.timer = null;
  entry.remaining = Math.max(0, entry.remaining - (Date.now() - entry.startedAt));
}

function resume(entry) {
  if (!entry.paused || entry.done) return;
  entry.paused = false;
  startTimer(entry, entry.remaining);
}

/**
 * message  the headline — always names what changed.
 * opts     { kind, detail, timeout, undo:{label,onUndo,onExpire},
 *            action:{label,onClick,variant}, dismissible, retry:fn }
 */
export function toast(message, opts) {
  var o = opts || {};
  var kind = o.kind || 'success';
  var timeout = o.timeout === undefined ? TIMEOUTS[kind] : o.timeout;
  if (o.undo && o.timeout === undefined) timeout = UNDO_MS;

  var el = h('div.toast', { 'class': 'toast--' + kind });
  if (kind === 'error') el.setAttribute('role', 'alert');

  var iconBox = h('span.toast__icon', icon(ICONS[kind] || 'info', { size: 'sm' }));
  var titleEl = h('strong.toast__title', { text: message });
  var detailEl = h('span.toast__detail');
  if (o.detail) detailEl.textContent = o.detail;
  else detailEl.hidden = true;

  var text = h('div.toast__text', titleEl, detailEl);
  el.appendChild(iconBox);
  el.appendChild(text);

  var entry = {
    el: el, done: false, timer: null, paused: false,
    remaining: timeout, startedAt: 0,
    onExpire: o.undo ? o.undo.onExpire : null
  };

  if (o.undo) {
    var undoBtn = h('button.btn.btn--secondary.toast__action', { type: 'button' },
      icon('undo', { size: 'sm', 'class': 'btn__icon' }),
      h('span.btn__label', { text: o.undo.label || 'Undo' }));
    undoBtn.addEventListener('click', function () {
      entry.onExpire = null;                  /* undo cancels the deadline */
      dismiss(entry, false);
      if (o.undo.onUndo) o.undo.onUndo();
    });
    el.appendChild(undoBtn);
  } else if (o.action) {
    var actBtn = h('button.btn.toast__action',
      { type: 'button', 'class': 'btn--' + (o.action.variant || 'secondary') },
      h('span.btn__label', { text: o.action.label }));
    actBtn.addEventListener('click', function () {
      dismiss(entry, false);
      if (o.action.onClick) o.action.onClick();
    });
    el.appendChild(actBtn);
  }

  if (o.retry) {
    var retryBtn = h('button.btn.btn--secondary.toast__action', { type: 'button' },
      icon('refresh', { size: 'sm', 'class': 'btn__icon' }),
      h('span.btn__label', { text: 'Retry' }));
    retryBtn.addEventListener('click', function () {
      dismiss(entry, false);
      o.retry();
    });
    el.appendChild(retryBtn);
  }

  /* Anything that does not auto-dismiss must be dismissible by hand. */
  if (!timeout || o.dismissible) {
    var closeBtn = h('button.icon-btn.toast__close',
      { type: 'button', 'aria-label': 'Dismiss notification' },
      icon('x', { size: 'sm' }));
    closeBtn.addEventListener('click', function () { dismiss(entry, false); });
    el.appendChild(closeBtn);
  }

  if (o.undo) el.appendChild(h('span.toast__timer', { 'aria-hidden': 'true' }));
  else if (kind === 'progress') el.appendChild(h('span.toast__progress', { 'aria-hidden': 'true' }));

  el.addEventListener('mouseenter', function () { pause(entry); });
  el.addEventListener('mouseleave', function () { resume(entry); });
  el.addEventListener('focusin', function () { pause(entry); });
  el.addEventListener('focusout', function () { resume(entry); });

  ensureHost().appendChild(el);
  live.push(entry);
  /* A fourth toast collapses the oldest — but never one holding a deadline
     the user has not seen out. */
  while (live.length > MAX_VISIBLE) {
    var oldest = live[0];
    dismiss(oldest, true);
  }

  startTimer(entry, timeout);

  return {
    el: el,
    close: function (fireExpire) { dismiss(entry, !!fireExpire); },
    update: function (patch) {
      if (!patch) return;
      if (patch.title !== undefined) titleEl.textContent = patch.title;
      if (patch.detail !== undefined) {
        detailEl.textContent = patch.detail;
        detailEl.hidden = !patch.detail;
      }
      if (patch.kind && patch.kind !== kind) {
        el.classList.remove('toast--' + kind);
        kind = patch.kind;
        el.classList.add('toast--' + kind);
        clear(iconBox).appendChild(icon(ICONS[kind] || 'info', { size: 'sm' }));
      }
    }
  };
}

toast.success = function (msg, opts) { return toast(msg, Object.assign({ kind: 'success' }, opts)); };
toast.info = function (msg, opts) { return toast(msg, Object.assign({ kind: 'info' }, opts)); };
toast.danger = function (msg, opts) { return toast(msg, Object.assign({ kind: 'danger' }, opts)); };
toast.progress = function (msg, opts) { return toast(msg, Object.assign({ kind: 'progress' }, opts)); };

/** A failure that must not disappear on its own. Names the cause; offers Retry. */
toast.error = function (msg, opts) {
  return toast(msg, Object.assign({ kind: 'error', timeout: 0 }, opts));
};

/** Fire every pending deadline right now — used on pagehide. */
toast.flush = function () {
  var pending = live.slice();
  for (var i = 0; i < pending.length; i++) dismiss(pending[i], true);
};

toast.clear = function () {
  var pending = live.slice();
  for (var i = 0; i < pending.length; i++) dismiss(pending[i], false);
};

export default toast;
