/* ============================================================================
   views/live.js — the operator surface. Route /app/live.

   THE ONE THING THIS FILE EXISTS FOR: a dead stream must never read as live.

   An <img> pointed at a multipart MJPEG endpoint freezes on its last decoded
   frame when the connection drops. Nothing in the DOM changes. The picture is
   still there, still sharp, still wrong. So this view never trusts the <img>.
   It cross-checks four independent signals:

     1. the server's own opinion  — /api/cameras state + frame_age, polled
     2. the <img> error event     — the socket died loudly
     3. <img> load recency        — WebKit fires load per MJPEG part; a long
                                    gap is a stall even when nothing errored
     4. a stall deadline          — no part and no error for STALL_MS is
                                    treated as a drop and forces a reconnect

   Any of them going bad applies the degraded treatment (.cam--stale /
   .cam--offline: desaturation, hazard hatch, freeze scanline veil, an
   on-media LAST FRAME … AGO pill) and starts a backoff reconnect with a live
   attempt counter. The picture is never left looking healthy.

   The second thing this file exists for: a lost PTZ stop leaves a camera
   slewing until the server's 10s dead-man fires. Every path that can end a
   jog — pointerup, pointercancel, touchcancel, lostpointercapture, keyup,
   Escape, window blur, visibilitychange, pagehide, unmount — routes through
   stopJog(), and a held jog re-sends its move every REPEAT_MS so the backstop
   can later be tightened.

   iOS 15 floor: var/function style, no optional chaining, no ??, no top-level
   await, no .at().
   ========================================================================= */

import { h, on, delegate, keyedList, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { store } from '../core/store.js';
import { router } from '../core/router.js';
import { api } from '../core/api.js';
import { toast } from '../core/toast.js';
import { dialog, sheet } from '../core/overlay.js';
import { shortAgo, plural } from '../core/format.js';

/* --- tuning -------------------------------------------------------------- */

var CAMERAS_MS = 3000;       /* /api/cameras poll */
var MONITOR_MS = 5000;       /* /api/monitor poll */
var POSITION_MS = 2500;      /* /ptz/{id}/position poll */
var SNAPSHOT_MS = 6000;      /* filmstrip stills */
var TICK_MS = 1000;          /* age pills + retry countdowns */
var REPEAT_MS = 1800;        /* re-send a held move under the dead-man */
var STALL_MS = 12000;        /* no MJPEG part and no error for this long */
var VERIFY_MS = 45000;       /* quiet-engine socket proof-of-life */
var WATCH_MS = 2000;         /* how often the watchdog re-examines a card */
var STALE_AGE = 5;           /* seconds of frame_age before "stale" */
var DEAD_AGE = 30;           /* seconds of frame_age before "offline" */
var BACKOFF = [1000, 2000, 4000, 8000, 15000, 30000];
var PULSE_MS = 420;          /* click-to-centre move pulse */
var BLANK = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
var SPEEDS = { slow: 0.25, normal: 0.55, fast: 0.9 };

/* --- tiny utils ---------------------------------------------------------- */

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
function pad2(n) { return n < 10 ? '0' + n : String(n); }
function num(v) { return typeof v === 'number' && isFinite(v) ? v : null; }
function wallClock(d) {
  return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
}
function fixed(v, places) {
  var n = num(v);
  return n === null ? '—' : n.toFixed(places === undefined ? 2 : places);
}
function setText(el, text) {
  var s = String(text === null || text === undefined ? '' : text);
  if (el.textContent !== s) el.textContent = s;
}
function toggleClass(el, cls, want) {
  if (!el) return;
  if (want) el.classList.add(cls); else el.classList.remove(cls);
}
/* Engines disagree about MJPEG: WebKit fires `load` for every part, Blink
   fires it once per attachment. Timing the gap between loads is therefore
   only a valid stall signal on an engine that has PROVED it is chatty — one
   that has delivered a second load on a single attachment. Until then the
   watchdog leans on the server's frame_age and on a periodic re-attach. */
var CHATTY = false;

/**
 * Put exactly `nodes`, in this order, as the children of `parent`, moving
 * only what is actually out of place. Layout changes therefore re-parent
 * live <img> elements instead of rebuilding them, so a running MJPEG socket,
 * the focus ring and a half-finished gesture all survive a mode switch.
 */
function placeChildren(parent, nodes) {
  var cursor = parent.firstChild;
  for (var i = 0; i < nodes.length; i++) {
    var n = nodes[i];
    if (cursor === n) { cursor = cursor.nextSibling; continue; }
    parent.insertBefore(n, cursor);
  }
  var child = cursor;
  while (child) {
    var next = child.nextSibling;
    if (nodes.indexOf(child) < 0) parent.removeChild(child);
    child = next;
  }
}

/* Failure noise control: a poll that fails every 3s must not emit a toast
   every 3s. Inline error text is always updated; the toast is throttled. */
var lastShout = Object.create(null);
function shout(key, message, err, opts) {
  var now = Date.now();
  if (lastShout[key] && now - lastShout[key] < 30000) return null;
  lastShout[key] = now;
  var o = opts || {};
  o.detail = api.describe(err);
  return toast.error(message, o);
}

/* ========================================================================
   Small components. Each returns { el, set } and patches in place, because
   a refresh must never blow away focus or a half-finished gesture.
   ===================================================================== */

function makeReadout(label, opts) {
  var o = opts || {};
  var valueEl = h('span.readout__value', { text: '—' });
  var unitEl = o.unit ? h('span.readout__unit', { text: o.unit }) : null;
  var meterEl = null;
  var srEl = null;
  var el = h('div.readout',
    h('span.readout__label', { text: label }),
    h('span', valueEl, unitEl));
  if (o.meter) {
    meterEl = h('span.meter', { style: { '--n': '0', '--of': String(o.of || 8) } });
    srEl = h('span.visually-hidden');
    el.appendChild(h('span', meterEl, srEl));
  }
  var mods = ['readout--live', 'readout--warn', 'readout--danger', 'readout--stale'];
  return {
    el: el,
    set: function (value, mod, meterN, srText) {
      setText(valueEl, value);
      for (var i = 0; i < mods.length; i++) el.classList.remove(mods[i]);
      if (mod) el.classList.add('readout--' + mod);
      if (meterEl) {
        meterEl.style.setProperty('--n', String(meterN || 0));
        meterEl.classList.remove('meter--live', 'meter--stale', 'meter--danger', 'meter--accent');
        if (mod === 'live') meterEl.classList.add('meter--live');
        else if (mod === 'warn') meterEl.classList.add('meter--stale');
        else if (mod === 'danger') meterEl.classList.add('meter--danger');
        else meterEl.classList.add('meter--accent');
        setText(srEl, srText || (label + ' ' + value));
      }
    }
  };
}

function makeStatusPill(onMedia) {
  var dot = h('span.status__dot', { 'aria-hidden': 'true' });
  var labelEl = h('span.status__label', { text: 'Connecting' });
  var ageEl = h('span.status__age');
  ageEl.hidden = true;
  var el = h('span.status', { 'class': onMedia ? 'status--on-media' : null },
    dot, labelEl, ageEl);
  var mods = ['status--live', 'status--stale', 'status--offline',
    'status--recording', 'status--reconnecting', 'status--unknown'];
  return {
    el: el,
    set: function (kind, word, age) {
      for (var i = 0; i < mods.length; i++) el.classList.remove(mods[i]);
      el.classList.add('status--' + kind);
      setText(labelEl, word);
      if (age) { setText(ageEl, age); ageEl.hidden = false; }
      else ageEl.hidden = true;
    }
  };
}

function makeSwitchRow(title, hint, onToggle) {
  var stateEl = h('span.switch-row__state', { text: 'Off' });
  var el = h('button.switch-row', {
    type: 'button', role: 'switch', 'aria-checked': 'false'
  },
    h('span.switch-row__text',
      h('span.switch-row__title', { text: title }),
      hint ? h('span.switch-row__hint', { text: hint }) : null),
    stateEl,
    h('span.switch', h('span.switch__knob')));
  el.addEventListener('click', function () {
    if (el.getAttribute('aria-busy') === 'true') return;
    if (el.getAttribute('aria-disabled') === 'true') return;
    onToggle(el.getAttribute('aria-checked') !== 'true');
  });
  return {
    el: el,
    set: function (checked, busy) {
      el.setAttribute('aria-checked', checked ? 'true' : 'false');
      setText(stateEl, busy ? 'Working' : (checked ? 'On' : 'Off'));
      if (busy) el.setAttribute('aria-busy', 'true');
      else el.removeAttribute('aria-busy');
    },
    disable: function (why) {
      el.setAttribute('aria-disabled', 'true');
      setText(stateEl, why || 'N/A');
    },
    enable: function () { el.removeAttribute('aria-disabled'); }
  };
}

/**
 * Pointer + keyboard slider. role=slider with real arrow-key support; the
 * knob's left is a percentage of the .slider box, so no .slider__output is
 * placed inside it (that would shorten the box and skew the knob).
 */
function makeSlider(opts) {
  var o = opts || {};
  var min = o.min === undefined ? 0 : o.min;
  var max = o.max === undefined ? 1 : o.max;
  var step = o.step || 0.01;
  var value = o.value === undefined ? min : o.value;

  var el = h('div.slider', {
    role: 'slider', tabIndex: 0,
    'aria-label': o.label || 'Slider',
    'aria-valuemin': String(min), 'aria-valuemax': String(max)
  }, h('div.slider__rail', h('div.slider__fill')), h('div.slider__knob'));

  function render() {
    var frac = max === min ? 0 : (value - min) / (max - min);
    el.style.setProperty('--v', String(clamp(frac, 0, 1)));
    el.setAttribute('aria-valuenow', String(value));
    el.setAttribute('aria-valuetext', o.format ? o.format(value) : String(value));
  }
  function quantise(v) {
    var stepped = Math.round((v - min) / step) * step + min;
    return clamp(Math.round(stepped * 1e6) / 1e6, min, max);
  }
  function fromClient(clientX) {
    var r = el.getBoundingClientRect();
    if (!r.width) return value;
    return quantise(min + ((clientX - r.left) / r.width) * (max - min));
  }
  function commitInput(v) {
    if (v === value) return;
    value = v;
    render();
    if (o.onInput) o.onInput(value);
  }

  var dragging = false;
  el.addEventListener('pointerdown', function (ev) {
    if (el.getAttribute('aria-disabled') === 'true') return;
    dragging = true;
    el.classList.add('slider--dragging');
    try { el.setPointerCapture(ev.pointerId); } catch (e) { /* Safari 15 */ }
    el.focus();
    if (o.onStart) o.onStart(value);
    commitInput(fromClient(ev.clientX));
    ev.preventDefault();
  });
  el.addEventListener('pointermove', function (ev) {
    if (!dragging) return;
    commitInput(fromClient(ev.clientX));
  });
  function endDrag() {
    if (!dragging) return;
    dragging = false;
    el.classList.remove('slider--dragging');
    if (o.onCommit) o.onCommit(value);
  }
  el.addEventListener('pointerup', endDrag);
  el.addEventListener('pointercancel', endDrag);
  el.addEventListener('lostpointercapture', endDrag);
  el.addEventListener('touchcancel', endDrag);
  el.addEventListener('keydown', function (ev) {
    var k = ev.key;
    var big = (max - min) / 10;
    var next = value;
    if (k === 'ArrowRight' || k === 'ArrowUp') next = quantise(value + (ev.shiftKey ? step : big));
    else if (k === 'ArrowLeft' || k === 'ArrowDown') next = quantise(value - (ev.shiftKey ? step : big));
    else if (k === 'Home') next = min;
    else if (k === 'End') next = max;
    else if (k === 'PageUp') next = quantise(value + big);
    else if (k === 'PageDown') next = quantise(value - big);
    else return;
    ev.preventDefault();
    if (o.onStart) o.onStart(value);
    commitInput(next);
    if (o.onCommit) o.onCommit(next);
  });

  render();
  return {
    el: el,
    get: function () { return value; },
    set: function (v) { if (dragging) return; value = clamp(v, min, max); render(); },
    isDragging: function () { return dragging; }
  };
}

function makeSegmented(label, options, initial, onPick) {
  var el = h('div.seg', { role: 'group', 'aria-label': label });
  var btns = [];
  options.forEach(function (opt) {
    var b = h('button.seg__btn', {
      type: 'button',
      'aria-pressed': opt.value === initial ? 'true' : 'false'
    }, opt.label);
    b.addEventListener('click', function () {
      btns.forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      onPick(opt.value);
    });
    btns.push(b);
    el.appendChild(b);
  });
  return { el: el };
}

/* ========================================================================
   View state
   ===================================================================== */

var S = null;

function newState() {
  return {
    root: null,
    disposers: [],
    timers: [],
    aborter: (typeof AbortController === 'function') ? new AbortController() : null,
    cams: [],              /* raw /api/cameras rows */
    tz: '',
    monitor: Object.create(null),
    cards: new Map(),      /* id -> card */
    primary: null,
    mode: 'focus',         /* 'focus' | 'grid' */
    desktop: false,
    hidden: false,
    dead: false,
    jog: null,
    speed: 'normal',
    wall: null, strip: null, operator: null, toolbar: null,
    fallbackEl: null, fallbackKind: null, fallbackBody: null,
    liveRegion: null,
    layoutBtn: null, streamsBtn: null,
    mq: null, mqOff: null,
    camsError: null,
    lastSummary: null,
    tickTimer: null, camTimer: null, monTimer: null, posTimer: null, snapTimer: null
  };
}

function reg(off) { if (off) S.disposers.push(off); return off; }
function every(fn, ms) {
  var id = window.setInterval(fn, ms);
  S.timers.push(id);
  return id;
}
function later(fn, ms) {
  var id = window.setTimeout(function () {
    var i = S.timers.indexOf(id);
    if (i >= 0) S.timers.splice(i, 1);
    fn();
  }, ms);
  S.timers.push(id);
  return id;
}
function cancel(id) {
  if (id === null || id === undefined) return;
  window.clearTimeout(id);
  window.clearInterval(id);
  var i = S.timers.indexOf(id);
  if (i >= 0) S.timers.splice(i, 1);
}
function signal() { return S.aborter ? S.aborter.signal : undefined; }

/* ========================================================================
   PTZ command plumbing
   ===================================================================== */

function ptzIdFor(cam) {
  /* cam1 (wide) drives cam2's head in this rig; ptz_target names it. Manual
     control has to follow the same wire or the buttons lie. */
  return cam.ptz_target || cam.id;
}

function sendMove(card, vec) {
  api.ptz.move(card.ptzId, vec, { signal: signal(), timeout: 6000 })
    .catch(function (err) {
      if (api.isAbort(err)) return;
      shout('ptz-move-' + card.ptzId, 'PTZ move failed on ' + card.ptzId, err);
    });
}

function sendStop(card) {
  /* keepalive so a stop issued during pagehide still leaves the machine. */
  api.ptz.stop(card.ptzId, { timeout: 6000, keepalive: true })
    .catch(function (err) {
      if (api.isAbort(err)) return;
      shout('ptz-stop-' + card.ptzId, 'PTZ stop failed on ' + card.ptzId, err, {
        detail: 'The camera may still be moving. Try Stop again.'
      });
    });
}

function speedFactor(fine) {
  var f = SPEEDS[S.speed] || SPEEDS.normal;
  return fine ? f * 0.3 : f;
}

/**
 * Start a held jog. Any previously held jog is stopped first, so two pointers
 * can never leave one of them latched.
 */
function startJog(card, vec, btn, fine) {
  if (card.controlDown) return;
  stopJog();
  var f = speedFactor(fine);
  var v = {
    pan: clamp((vec.pan || 0) * f, -1, 1),
    tilt: clamp((vec.tilt || 0) * f, -1, 1),
    zoom: clamp((vec.zoom || 0) * f, -1, 1)
  };
  S.jog = { card: card, vec: v, btn: btn || null, repeat: null };
  if (btn) btn.classList.add('is-jogging');
  card.stageEl.classList.add('is-jogging');
  sendMove(card, v);
  S.jog.repeat = every(function () {
    if (S.jog) sendMove(S.jog.card, S.jog.vec);
  }, REPEAT_MS);
  announce(card.name + ' moving');
}

function stopJog() {
  if (!S || !S.jog) return;
  var j = S.jog;
  S.jog = null;
  cancel(j.repeat);
  if (j.btn) j.btn.classList.remove('is-jogging');
  if (j.card && j.card.stageEl) j.card.stageEl.classList.remove('is-jogging');
  sendStop(j.card);
}

/** A bounded nudge: move, then stop. Used by click-to-centre and taps. */
function pulse(card, vec, ms) {
  stopJog();
  sendMove(card, vec);
  card.stageEl.classList.add('is-jogging');
  cancel(card.pulseTimer);
  card.pulseTimer = later(function () {
    card.stageEl.classList.remove('is-jogging');
    sendStop(card);
  }, ms || PULSE_MS);
}

function announce(text) {
  if (S && S.liveRegion) setText(S.liveRegion, text);
}

/* ========================================================================
   Camera card
   ===================================================================== */

function buildCard(cam) {
  var card = {
    id: cam.id,
    name: cam.name || cam.id,
    cam: cam,
    ptzId: ptzIdFor(cam),
    /* stream watchdog */
    attached: false,
    lastLoad: 0,
    loads: 0,
    verifying: false,
    imgError: false,
    attempt: 0,
    nextTryAt: 0,
    retryTimer: null,
    stallTimer: null,
    pulseTimer: null,
    faultCause: '',
    state: 'unknown',
    frameAge: null,
    /* ptz */
    position: null,
    posError: null,
    presets: [],
    presetsError: null,
    mode: null,
    dragStart: null,
    controlDown: false
  };

  /* --- head ------------------------------------------------------------ */
  card.nameEl = h('span', { text: card.name });
  card.idEl = h('span.cam__id');
  card.handoff = h('span.cam__handoff');
  card.handoff.hidden = true;
  card.fsBtn = h('button.icon-btn', {
    type: 'button', 'aria-label': 'Fullscreen ' + card.name + ' (F)'
  }, icon('external'));
  card.fsBtn.addEventListener('click', function () { toggleFullscreen(card); });

  card.headStatus = makeStatusPill(false);

  card.head = h('header.cam__head',
    h('h2.cam__name', card.nameEl, card.idEl),
    h('div.cam__headend', card.handoff, card.headStatus.el, card.fsBtn));

  /* --- stage ----------------------------------------------------------- */
  card.imgEl = h('img.frame__img', {
    alt: '', decoding: 'async', draggable: false
  });
  card.imgEl.addEventListener('load', function () {
    /* Tearing a socket down assigns a blank data URI, and THAT fires load
       too. Honouring it would clear the fault cause and cancel the pending
       reconnect, so a dropped stream would sit dark for ever. Only a load
       that happens while we believe we are attached counts. */
    if (!card.attached) return;
    card.loads += 1;
    if (card.loads > 1) CHATTY = true;
    card.lastLoad = Date.now();
    card.imgError = false;
    card.verifying = false;
    card.faultCause = '';
    if (card.attempt) {
      card.attempt = 0;
      cancel(card.retryTimer);
      card.retryTimer = null;
    }
    armStall(card);
    paintCard(card);
  });
  card.imgEl.addEventListener('error', function () {
    if (!card.attached) return;
    card.imgError = true;
    onStreamDrop(card, 'The browser could not read the MJPEG stream.');
  });

  card.mediaStatus = makeStatusPill(true);
  card.clockEl = h('span.mpill.mpill--strong', { text: '--:--:--' });
  card.rateEl = h('span.mpill', { text: '—' });
  card.ageEl = h('div.cam__age');
  card.ageEl.hidden = true;

  card.crossEl = h('div.ptzstage__cross');
  card.crossEl.hidden = true;
  card.marqueeEl = h('div.ptzstage__marquee');
  card.marqueeEl.hidden = true;
  card.reticleEl = h('div.ptzstage__reticle',
    h('span.ptzstage__reticle-label'));
  card.reticleEl.hidden = true;
  card.reticleLabel = card.reticleEl.firstChild;

  card.frameEl = h('div.frame',
    h('div.frame__film.film--day'),
    card.imgEl,
    h('div.frame__scrim'),
    h('div.frame__tl', card.mediaStatus.el),
    h('div.frame__tr', h('span.mpill', { text: card.name })),
    h('div.frame__bl', card.clockEl),
    h('div.frame__br', card.rateEl));

  card.stageEl = h('div.cam__stage.ptzstage', {
    role: 'group',
    tabIndex: 0,
    'aria-label': card.name + ' video. Click to centre, drag to frame, arrow keys to pan.'
  },
    card.frameEl,
    h('div.cam__veil'),
    card.ageEl,
    h('div.ptzstage__layer', card.crossEl, card.marqueeEl, card.reticleEl),
    h('div.ptzstage__keys',
      h('kbd', '↑↓←→ pan'), h('kbd', '+ / − zoom'),
      h('kbd', '1–9 preset'), h('kbd', '0 stop'), h('kbd', 'F full'), h('kbd', 'S save')));

  wireStage(card);

  /* --- telemetry ------------------------------------------------------- */
  card.rAge = makeReadout('Frame age');
  card.rBuffer = makeReadout('Buffer', { meter: true, of: 10, unit: 's' });
  card.rTracks = makeReadout('Tracks');
  card.rTop = makeReadout('Top', { meter: true, of: 8 });
  card.rEvent = makeReadout('Event');
  card.telemetryEl = h('div.cam__telemetry', {
    role: 'group', 'aria-label': card.name + ' telemetry'
  },
    card.rAge.el, card.rBuffer.el, card.rTracks.el, card.rTop.el, card.rEvent.el);

  /* --- fault block ----------------------------------------------------- */
  card.faultTitle = h('h3.cam__fault-title', { text: 'Stream lost' });
  card.faultCauseEl = h('p.cam__fault-cause');
  card.faultRetryEl = h('p.cam__fault-retry');
  card.reconnectBtn = h('button.btn.btn--primary', { type: 'button' },
    icon('refresh', { size: 'sm', 'class': 'btn__icon' }),
    h('span.btn__label', 'Reconnect now'));
  card.reconnectBtn.addEventListener('click', function () { reconnectNow(card); });
  card.logBtn = h('button.btn.btn--secondary', { type: 'button' },
    icon('list', { size: 'sm', 'class': 'btn__icon' }),
    h('span.btn__label', 'View log'));
  card.logBtn.addEventListener('click', function () {
    router.go('/monitor', { camera: card.id, tab: 'logs' });
  });
  card.faultEl = h('div.cam__fault', { role: 'group' },
    h('div.cam__fault-head',
      h('span.cam__spinner', { 'aria-hidden': 'true' }),
      h('div', card.faultTitle, card.faultCauseEl)),
    card.faultRetryEl,
    h('div.cam__fault-actions', card.reconnectBtn, card.logBtn));
  card.faultEl.hidden = true;

  /* --- save clip ------------------------------------------------------- */
  card.saveBtn = h('button.btn.btn--secondary.btn--lg.btn--block', { type: 'button' },
    icon('download', { 'class': 'btn__icon' }),
    h('span.btn__label', 'Save last 30 s'),
    h('span.btn__spinner', h('span.spinner')));
  card.saveBtn.addEventListener('click', function () { saveClip(card); });

  /* --- ptz cluster ----------------------------------------------------- */
  buildPtz(card, cam);

  card.noControl = h('p.cam__nocontrol', {
    text: 'Controls unavailable while the stream is down.'
  });
  card.noControl.hidden = true;

  card.bodyEl = h('div.cam__body',
    card.telemetryEl, card.faultEl, card.saveBtn,
    card.ptzEl || null, card.noControl);

  card.el = h('article.cam', {
    'data-cam': cam.id,
    'aria-label': card.name + ' camera'
  }, card.head, card.stageEl, card.bodyEl);

  return card;
}

/* ------------------------------------------------------------------ PTZ UI */

var DIRS = [
  { key: 'nw', pan: -1, tilt: 1, rot: -45, label: 'Pan left and tilt up' },
  { key: 'n', pan: 0, tilt: 1, rot: 0, label: 'Tilt up' },
  { key: 'ne', pan: 1, tilt: 1, rot: 45, label: 'Pan right and tilt up' },
  { key: 'w', pan: -1, tilt: 0, rot: -90, label: 'Pan left' },
  { key: 'stop', pan: 0, tilt: 0, rot: 0, label: 'Stop all movement' },
  { key: 'e', pan: 1, tilt: 0, rot: 90, label: 'Pan right' },
  { key: 'sw', pan: -1, tilt: -1, rot: -135, label: 'Pan left and tilt down' },
  { key: 's', pan: 0, tilt: -1, rot: 180, label: 'Tilt down' },
  { key: 'se', pan: 1, tilt: -1, rot: 135, label: 'Pan right and tilt down' }
];

function buildPtz(card, cam) {
  if (!cam.has_ptz) { card.ptzEl = null; return; }

  /* d-pad ------------------------------------------------------------- */
  card.pad = h('div.ptz__pad', { role: 'group', 'aria-label': 'Pan and tilt' });
  card.jogBtns = Object.create(null);
  DIRS.forEach(function (d) {
    var glyph;
    if (d.key === 'stop') glyph = icon('square', { size: 'sm' });
    else {
      glyph = icon('chevron-up');
      glyph.style.transform = 'rotate(' + d.rot + 'deg)';
    }
    var b = h('button.ptz__jog', {
      type: 'button',
      'class': d.key === 'stop' ? 'ptz__home' : null,
      'data-dir': d.key,
      'aria-label': d.label + (d.key === 'stop' ? ' (0)' : '')
    }, glyph);
    card.jogBtns[d.key] = b;
    card.pad.appendChild(b);
  });
  wirePad(card);

  /* zoom -------------------------------------------------------------- */
  card.zoomOut = h('button.ptz__zoombtn', {
    type: 'button', 'data-zoom': '-1', 'aria-label': 'Zoom out (hold)'
  }, icon('minus', { size: 'sm' }));
  card.zoomIn = h('button.ptz__zoombtn', {
    type: 'button', 'data-zoom': '1', 'aria-label': 'Zoom in (hold)'
  }, icon('plus', { size: 'sm' }));
  card.zoomOutput = h('output', { text: '—' });
  card.zoomSlider = makeSlider({
    label: 'Zoom', min: 0, max: 1, step: 0.01, value: 0,
    format: function (v) { return Math.round(v * 100) + '%'; },
    onStart: function (v) { card.zoomFrom = v; },
    onInput: function (v) {
      setText(card.zoomOutput, Math.round(v * 100) + '%');
      var from = card.zoomFrom === undefined ? v : card.zoomFrom;
      var dir = v > from ? 1 : (v < from ? -1 : 0);
      if (dir === 0) return;
      /* The head takes a velocity, not a position, so dragging the slider is
         a held zoom in the drag's direction and release stops it. The move is
         re-sent under REPEAT_MS for the same reason a held d-pad is: so the
         server's dead-man can be tightened without stranding a long drag. */
      var now = Date.now();
      if (card.zoomDir !== dir || now - (card.zoomSentAt || 0) > REPEAT_MS) {
        card.zoomDir = dir;
        card.zoomSentAt = now;
        sendMove(card, { pan: 0, tilt: 0, zoom: clamp(dir * speedFactor(), -1, 1) });
      }
    },
    onCommit: function () {
      var wasMoving = !!card.zoomDir;
      card.zoomDir = 0;
      card.zoomSentAt = 0;
      card.zoomFrom = undefined;
      if (wasMoving) sendStop(card);
    }
  });
  card.zoomRow = h('div.ptz__zoomrow',
    card.zoomOut, card.zoomSlider.el, card.zoomIn, card.zoomOutput);
  wireHold(card, card.zoomOut, function () {
    return { pan: 0, tilt: 0, zoom: -1 };
  });
  wireHold(card, card.zoomIn, function () {
    return { pan: 0, tilt: 0, zoom: 1 };
  });

  /* speed ------------------------------------------------------------- */
  card.speedSeg = makeSegmented('Jog speed', [
    { value: 'slow', label: 'Slow' },
    { value: 'normal', label: 'Normal' },
    { value: 'fast', label: 'Fast' }
  ], S ? S.speed : 'normal', function (v) { S.speed = v; });

  /* presets ----------------------------------------------------------- */
  card.presetsEl = h('div.ptz__presets', { role: 'group', 'aria-label': 'Presets' });
  card.presetNote = h('p.t-xs.t-3', { text: 'Loading presets…' });
  card.savePresetBtn = h('button.chip.chip--tonal', { type: 'button' },
    icon('bookmark', { size: 'sm' }), h('span', 'Save preset'));
  card.savePresetBtn.addEventListener('click', function () { savePresetFlow(card); });
  reg(delegate(card.presetsEl, 'click', '[data-preset]', function (ev, node) {
    gotoPreset(card, node.getAttribute('data-preset'), node.getAttribute('data-name'));
  }));

  /* switches ---------------------------------------------------------- */
  card.trackSwitch = makeSwitchRow('Auto-track', 'Follow detections with the head',
    function (next) { setMode(card, 'track', next); });
  card.patrolSwitch = makeSwitchRow('Patrol', 'Cycle the patrol presets when idle',
    function (next) { setMode(card, 'patrol', next); });
  card.debugSwitch = makeSwitchRow('PTZ debug logging', 'Verbose tracker logs on the server',
    function (next) { setDebug(card, next); });

  card.returnOut = h('span.readout__value', { text: '—' });
  card.returnSlider = makeSlider({
    label: 'Return to patrol after', min: 0, max: 300, step: 5, value: 5,
    format: function (v) { return v + ' seconds'; },
    onInput: function (v) { setText(card.returnOut, v + ' s'); },
    onCommit: function (v) { setReturnDelay(card, v); }
  });
  card.returnBlock = h('div.stack.stack--tight',
    h('div.row.row--between',
      h('span.readout__label', { text: 'Return to patrol' }),
      card.returnOut),
    card.returnSlider.el);

  /* banner + readout -------------------------------------------------- */
  card.banner = h('div.ptz__banner', icon('info', { size: 'sm' }), h('span'));
  card.banner.hidden = true;
  card.bannerText = card.banner.lastChild;

  card.readoutEl = h('div.ptz__readout');
  card.pEl = h('b', { text: '—' });
  card.tEl = h('b', { text: '—' });
  card.zEl = h('b', { text: '—' });
  card.posNote = h('span');
  card.readoutEl.appendChild(h('span', 'P ', card.pEl));
  card.readoutEl.appendChild(h('span', 'T ', card.tEl));
  card.readoutEl.appendChild(h('span', 'Z ', card.zEl));
  card.readoutEl.appendChild(card.posNote);

  /* calibration ------------------------------------------------------- */
  card.calibEl = buildCalibration(card, cam);

  card.ptzEl = h('div.ptz', { 'aria-label': 'PTZ controls for ' + card.ptzId },
    card.banner,
    card.pad,
    card.zoomRow,
    card.speedSeg.el,
    h('div.stack.stack--tight', card.presetsEl, card.presetNote),
    h('div.ptz__switches', card.trackSwitch.el, card.patrolSwitch.el, card.debugSwitch.el),
    card.returnBlock,
    card.readoutEl,
    card.calibEl || null);

  if (card.ptzId !== card.id) {
    card.banner.hidden = false;
    setText(card.bannerText, 'These controls drive ' + card.ptzId + ' — ' +
      card.id + ' is the wide camera that aims it.');
  }
}

function buildCalibration(card, cam) {
  /* Both of these physically slew the head for tens of seconds. They are
     confirmed first, then reported with a progress toast. */
  if (!cam.ptz_target) return null;
  var wide = cam.id;
  var zoom = cam.ptz_target;

  var visualBtn = h('button.btn.btn--secondary.btn--block', { type: 'button' },
    icon('sparkle', { size: 'sm', 'class': 'btn__icon' }),
    h('span.btn__label', 'Auto-calibrate PTZ'),
    h('span.btn__spinner', h('span.spinner')));
  var fovBtn = h('button.btn.btn--secondary.btn--block', { type: 'button' },
    icon('layers', { size: 'sm', 'class': 'btn__icon' }),
    h('span.btn__label', 'Calibrate zoom FOV'),
    h('span.btn__spinner', h('span.spinner')));

  visualBtn.addEventListener('click', function () {
    runCalibration(card, visualBtn, {
      title: 'Move the camera to calibrate?',
      body: 'This drives ' + zoom + ' through a 3×3 grid of positions and matches ' +
        'each view against ' + wide + '. The head will move for roughly a minute and ' +
        'tracking is unavailable until it finishes.',
      stakes: 'The camera physically moves. Nothing is recorded during the sweep.',
      endpoint: '/ptz/calibrate',
      body_json: { wide_camera_id: wide, zoom_camera_id: zoom, grid_size: 3 },
      running: 'Calibrating PTZ — sweeping the grid'
    });
  });
  fovBtn.addEventListener('click', function () {
    runCalibration(card, fovBtn, {
      title: 'Move the camera to map zoom FOV?',
      body: 'This steps ' + zoom + ' through 0 %, 50 % and 100 % zoom and records what ' +
        'part of ' + wide + ' each level sees. Expect around thirty seconds of movement.',
      stakes: 'The camera physically moves and the result overwrites zoom_fov_calibration.json.',
      endpoint: '/ptz/zoom-fov-calibrate',
      body_json: {
        wide_camera_id: wide, zoom_camera_id: zoom,
        zoom_levels: '0.0,0.5,1.0', settle_time: 2.0
      },
      running: 'Mapping zoom FOV — stepping zoom levels'
    });
  });

  return h('div.stack.stack--tight',
    h('span.overline.overline--strong', { text: 'Calibration' }),
    visualBtn, fovBtn);
}

/* ------------------------------------------------------------- PTZ actions */

function busy(btn, on) {
  if (!btn) return;
  if (on) btn.setAttribute('aria-busy', 'true');
  else btn.removeAttribute('aria-busy');
}

function runCalibration(card, btn, spec) {
  dialog({
    tone: 'danger',
    icon: 'alert',
    title: spec.title,
    body: spec.body,
    stakes: spec.stakes,
    actions: [
      { label: 'Cancel', value: false },
      { label: 'Move the camera', variant: 'danger-solid', value: true, focus: true }
    ]
  }).result.then(function (go) {
    if (!go) return;
    busy(btn, true);
    var t = toast.progress(spec.running, {
      timeout: 0, dismissible: true,
      detail: 'The head is moving. Leave it alone until this finishes.'
    });
    api.request('POST', spec.endpoint, { body: spec.body_json, timeout: 0, signal: signal() })
      .then(function (res) {
        t.close();
        busy(btn, false);
        var err = res && res.error;
        if (err) { toast.error('Calibration failed.', { detail: String(err) }); return; }
        toast.success('Calibration finished.', {
          detail: typeof res === 'string' ? res.slice(0, 200) : 'Parameters written to config.'
        });
        announce('Calibration finished for ' + card.id);
      })
      .catch(function (err) {
        t.close();
        busy(btn, false);
        if (api.isAbort(err)) return;
        toast.error('Calibration failed.', { detail: api.describe(err) });
      });
  });
}

function gotoPreset(card, token, name) {
  if (!token) return;
  api.ptz.gotoPreset(card.ptzId, token, { signal: signal(), timeout: 15000 })
    .then(function () {
      toast.success('Recalled ' + (name || token) + '.');
      announce('Recalled preset ' + (name || token));
    })
    .catch(function (err) {
      if (api.isAbort(err)) return;
      toast.error('Could not recall that preset.', { detail: api.describe(err) });
    });
}

function savePresetFlow(card) {
  var input = h('input.input', {
    type: 'text', maxlength: '48',
    placeholder: 'Feeder, North gate, …',
    'aria-label': 'Preset name'
  });
  var dlg = dialog({
    role: 'dialog',
    icon: 'bookmark',
    title: 'Save the current position',
    body: 'The head stays where it is; the name is what you will see on the chip.',
    content: function (box) {
      box.appendChild(h('label.field',
        h('span.field__label', { text: 'Preset name' }), input));
    },
    initialFocus: input,
    actions: [
      { label: 'Cancel', value: null },
      { label: 'Save preset', variant: 'primary', value: 'save' }
    ]
  });
  input.addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter') { ev.preventDefault(); dlg.close('save'); }
  });
  dlg.result.then(function (v) {
    if (v !== 'save') return;
    var name = (input.value || '').trim();
    if (!name) { toast.danger('A preset needs a name.'); return; }
    api.ptz.savePreset(card.ptzId, name, { signal: signal(), timeout: 15000 })
      .then(function () {
        toast.success('Saved preset “' + name + '”.');
        loadPresets(card, true);
      })
      .catch(function (err) {
        if (api.isAbort(err)) return;
        toast.error('Could not save that preset.', { detail: api.describe(err) });
      });
  });
}

function setMode(card, which, next) {
  var sw = which === 'track' ? card.trackSwitch : card.patrolSwitch;
  sw.set(next, true);
  var call = which === 'track'
    ? api.ptz.track(card.ptzId, next, { signal: signal(), timeout: 10000 })
    : api.ptz.patrol(card.ptzId, next, { signal: signal(), timeout: 10000 });
  call.then(function () {
    sw.set(next, false);
    if (which === 'track') applyTrackBanner(card, next);
    announce((which === 'track' ? 'Auto-track ' : 'Patrol ') + (next ? 'on' : 'off'));
    loadMode(card);
  }).catch(function (err) {
    if (api.isAbort(err)) return;
    sw.set(!next, false);
    toast.error('Could not change ' + which + ' on ' + card.ptzId + '.',
      { detail: api.describe(err) });
  });
}

function setDebug(card, next) {
  card.debugSwitch.set(next, true);
  api.ptz.setDebug(next, { signal: signal(), timeout: 10000 })
    .then(function () { card.debugSwitch.set(next, false); })
    .catch(function (err) {
      if (api.isAbort(err)) return;
      card.debugSwitch.set(!next, false);
      toast.error('Could not toggle PTZ debug logging.', { detail: api.describe(err) });
    });
}

function setReturnDelay(card, seconds) {
  api.ptz.returnDelay(card.ptzId, seconds, { signal: signal(), timeout: 10000 })
    .then(function () { toast.success('Return to patrol after ' + seconds + ' s.'); })
    .catch(function (err) {
      if (api.isAbort(err)) return;
      toast.error('Could not set the return delay.', { detail: api.describe(err) });
    });
}

function applyTrackBanner(card, on) {
  if (!card.banner) return;
  if (card.ptzId !== card.id) return;      /* the hand-off banner wins */
  card.banner.hidden = !on;
  if (on) setText(card.bannerText,
    'Auto-track is engaged. Manual jogs are advisory and the tracker will take the head back.');
}

function saveClip(card) {
  busy(card.saveBtn, true);
  var t = toast.progress('Saving the last 30 seconds from ' + card.name + '…', { timeout: 0 });
  api.saveClip(card.id, { signal: signal() })
    .then(function (res) {
      t.close();
      busy(card.saveBtn, false);
      var text = typeof res === 'string' ? res : (res && res.filename) || 'Clip saved.';
      var m = /Clip saved:\s*(.+)$/.exec(String(text));
      var filename = m ? m[1].trim() : null;
      toast.success('Clip saved from ' + card.name + '.', {
        detail: filename || String(text).slice(0, 160),
        action: filename ? {
          label: 'View', variant: 'secondary',
          onClick: function () { router.go('/recordings', { q: filename }); }
        } : null
      });
      announce('Clip saved from ' + card.name);
    })
    .catch(function (err) {
      t.close();
      busy(card.saveBtn, false);
      if (api.isAbort(err)) return;
      toast.error('Could not save a clip from ' + card.name + '.', {
        detail: api.describe(err),
        retry: function () { saveClip(card); }
      });
    });
}

/* ------------------------------------------------------- press-and-hold ---
   pointerdown starts; pointerup, pointercancel, touchcancel,
   lostpointercapture, Escape, blur and visibilitychange all stop. Nothing
   here relies on the server's 10 s dead-man.                              */

function wireHold(card, btn, vecFor) {
  btn.addEventListener('pointerdown', function (ev) {
    if (btn.getAttribute('aria-disabled') === 'true') return;
    ev.preventDefault();
    try { btn.setPointerCapture(ev.pointerId); } catch (e) { /* older WebKit */ }
    startJog(card, vecFor(ev), btn, ev.shiftKey);
  });
  btn.addEventListener('pointerup', stopJog);
  btn.addEventListener('pointercancel', stopJog);
  btn.addEventListener('lostpointercapture', stopJog);
  btn.addEventListener('touchcancel', stopJog);
  btn.addEventListener('touchend', function (ev) { ev.preventDefault(); stopJog(); });
  btn.addEventListener('contextmenu', function (ev) { ev.preventDefault(); });
  /* Keyboard: Space/Enter held is a jog, released is a stop. */
  btn.addEventListener('keydown', function (ev) {
    if (ev.key !== ' ' && ev.key !== 'Enter') return;
    ev.preventDefault();
    if (ev.repeat) return;
    startJog(card, vecFor(ev), btn, ev.shiftKey);
  });
  btn.addEventListener('keyup', function (ev) {
    if (ev.key !== ' ' && ev.key !== 'Enter') return;
    ev.preventDefault();
    stopJog();
  });
  btn.addEventListener('blur', stopJog);
}

function wirePad(card) {
  DIRS.forEach(function (d) {
    var btn = card.jogBtns[d.key];
    if (d.key === 'stop') {
      btn.addEventListener('click', function () {
        stopJog();
        sendStop(card);
        announce('Stopped ' + card.ptzId);
      });
      return;
    }
    wireHold(card, btn, function (ev) {
      return { pan: d.pan, tilt: d.tilt, zoom: 0 };
    });
  });
}

/* ------------------------------------------------- stage pointer surface -- */

function stagePoint(card, ev) {
  var r = card.frameEl.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  return {
    x: clamp((ev.clientX - r.left) / r.width, 0, 1),
    y: clamp((ev.clientY - r.top) / r.height, 0, 1),
    rect: r
  };
}

function flashCross(card, x, y) {
  card.crossEl.hidden = false;
  card.crossEl.style.setProperty('--x', (x * 100) + '%');
  card.crossEl.style.setProperty('--y', (y * 100) + '%');
  /* Restart the CSS animation without a reflow-heavy clone. */
  card.crossEl.style.animation = 'none';
  /* eslint-disable-next-line no-unused-expressions */
  card.crossEl.offsetHeight;
  card.crossEl.style.animation = '';
  cancel(card.crossTimer);
  card.crossTimer = later(function () { card.crossEl.hidden = true; }, 600);
}

function centreOn(card, x, y) {
  var dx = (x - 0.5) * 2;
  var dy = (y - 0.5) * 2;
  var f = speedFactor();
  var dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 0.02) return;
  pulse(card, {
    pan: clamp(dx * f, -1, 1),
    tilt: clamp(-dy * f, -1, 1),
    zoom: 0
  }, Math.round(clamp(180 + dist * 700, 180, 900)));
  flashCross(card, x, y);
  announce('Centring ' + card.ptzId);
}

function frameBox(card, box) {
  /* Centre on the box, then zoom in by however much of the frame it fills. */
  var cx = box.x + box.w / 2;
  var cy = box.y + box.h / 2;
  centreOn(card, cx, cy);
  var fill = Math.max(box.w, box.h);
  if (fill >= 0.85) return;
  var amount = clamp(1 - fill, 0.15, 1);
  later(function () {
    pulse(card, { pan: 0, tilt: 0, zoom: clamp(speedFactor(), 0.2, 1) },
      Math.round(300 + amount * 900));
  }, 950);
  announce('Framing a region on ' + card.ptzId);
}

function wireStage(card) {
  var el = card.stageEl;

  el.addEventListener('pointerdown', function (ev) {
    if (!card.cam.has_ptz) return;
    if (card.state === 'offline' || card.state === 'no-route') return;
    if (ev.button !== undefined && ev.button !== 0) return;
    var p = stagePoint(card, ev);
    if (!p) return;
    card.dragStart = { x: p.x, y: p.y, id: ev.pointerId, moved: false };
    try { el.setPointerCapture(ev.pointerId); } catch (e) { /* older WebKit */ }
    ev.preventDefault();
  });

  el.addEventListener('pointermove', function (ev) {
    if (!card.dragStart) return;
    var p = stagePoint(card, ev);
    if (!p) return;
    var dx = Math.abs(p.x - card.dragStart.x) * p.rect.width;
    var dy = Math.abs(p.y - card.dragStart.y) * p.rect.height;
    if (!card.dragStart.moved && (dx > 8 || dy > 8)) card.dragStart.moved = true;
    if (!card.dragStart.moved) return;
    var box = {
      x: Math.min(card.dragStart.x, p.x),
      y: Math.min(card.dragStart.y, p.y),
      w: Math.abs(p.x - card.dragStart.x),
      h: Math.abs(p.y - card.dragStart.y)
    };
    card.dragStart.box = box;
    card.marqueeEl.hidden = false;
    card.marqueeEl.style.setProperty('--x', (box.x * 100) + '%');
    card.marqueeEl.style.setProperty('--y', (box.y * 100) + '%');
    card.marqueeEl.style.setProperty('--w', (box.w * 100) + '%');
    card.marqueeEl.style.setProperty('--h', (box.h * 100) + '%');
  });

  function endStage(ev, aborted) {
    if (!card.dragStart) return;
    var start = card.dragStart;
    card.dragStart = null;
    card.marqueeEl.hidden = true;
    if (aborted) return;
    if (start.moved && start.box && start.box.w > 0.03 && start.box.h > 0.03) {
      frameBox(card, start.box);
      return;
    }
    var p = stagePoint(card, ev);
    if (p) centreOn(card, p.x, p.y);
  }
  el.addEventListener('pointerup', function (ev) { endStage(ev, false); });
  el.addEventListener('pointercancel', function (ev) { endStage(ev, true); });
  el.addEventListener('touchcancel', function (ev) { endStage(ev, true); });
  el.addEventListener('lostpointercapture', function (ev) { endStage(ev, true); });
  el.addEventListener('contextmenu', function (ev) {
    if (card.cam.has_ptz) ev.preventDefault();
  });
  el.addEventListener('dragstart', function (ev) { ev.preventDefault(); });

  /* keyboard jog ------------------------------------------------------- */
  el.addEventListener('keydown', function (ev) {
    if (ev.altKey || ev.metaKey || ev.ctrlKey) return;
    var k = ev.key;
    if (k === 'Escape') { stopJog(); return; }
    if (!card.cam.has_ptz) return;

    var vec = null;
    if (k === 'ArrowUp') vec = { pan: 0, tilt: 1, zoom: 0 };
    else if (k === 'ArrowDown') vec = { pan: 0, tilt: -1, zoom: 0 };
    else if (k === 'ArrowLeft') vec = { pan: -1, tilt: 0, zoom: 0 };
    else if (k === 'ArrowRight') vec = { pan: 1, tilt: 0, zoom: 0 };
    else if (k === '+' || k === '=') vec = { pan: 0, tilt: 0, zoom: 1 };
    else if (k === '-' || k === '_') vec = { pan: 0, tilt: 0, zoom: -1 };

    if (vec) {
      ev.preventDefault();
      if (ev.repeat) return;                 /* the jog is already running */
      card.heldKey = k;
      startJog(card, vec, null, ev.shiftKey);
      return;
    }
    if (k === '0') {
      ev.preventDefault();
      stopJog();
      sendStop(card);
      announce('Stopped ' + card.ptzId);
      return;
    }
    if (k >= '1' && k <= '9') {
      ev.preventDefault();
      var idx = parseInt(k, 10) - 1;
      var p = card.presets[idx];
      if (p) gotoPreset(card, p.token, p.name);
      else toast.info('No preset ' + k + ' on ' + card.ptzId + '.');
      return;
    }
    if (k === 'f' || k === 'F') { ev.preventDefault(); toggleFullscreen(card); return; }
    if (k === 's' || k === 'S') { ev.preventDefault(); saveClip(card); return; }
  });
  el.addEventListener('keyup', function (ev) {
    if (card.heldKey && ev.key === card.heldKey) {
      card.heldKey = null;
      stopJog();
    }
  });
  el.addEventListener('blur', function () {
    card.heldKey = null;
    card.dragStart = null;
    card.marqueeEl.hidden = true;
    stopJog();
  });
}

function toggleFullscreen(card) {
  var el = card.stageEl;
  var doc = document;
  var current = doc.fullscreenElement || doc.webkitFullscreenElement || null;
  if (current) {
    if (doc.exitFullscreen) doc.exitFullscreen();
    else if (doc.webkitExitFullscreen) doc.webkitExitFullscreen();
    return;
  }
  var req = el.requestFullscreen || el.webkitRequestFullscreen;
  if (!req) {
    toast.info('This browser will not put the video full screen.', {
      detail: 'On iPhone, rotate the device instead — the stage fills the landscape width.'
    });
    return;
  }
  try {
    var p = req.call(el);
    if (p && p.catch) p.catch(function (err) {
      toast.danger('Full screen was refused.', { detail: String(err && err.message || err) });
    });
  } catch (err) {
    toast.danger('Full screen was refused.', { detail: String(err && err.message || err) });
  }
}

/* ========================================================================
   The stream watchdog
   ===================================================================== */

function attachStream(card) {
  if (S.hidden || S.dead) return;
  if (!cardWantsStream(card)) return;
  card.attached = true;
  card.imgError = false;
  card.loads = 0;
  card.lastLoad = Date.now();
  card.imgEl.src = '/stream/' + encodeURIComponent(card.id) + '?_=' + Date.now();
  armStall(card);
  paintCard(card);
}

function detachStream(card) {
  card.attached = false;
  cancel(card.stallTimer); card.stallTimer = null;
  cancel(card.retryTimer); card.retryTimer = null;
  /* removeAttribute alone leaves some engines holding the socket; assigning a
     data URI guarantees the multipart request is torn down. */
  card.imgEl.removeAttribute('src');
  card.imgEl.src = BLANK;
}

/** Only the cameras actually on screen as full stages hold an MJPEG socket. */
function cardWantsStream(card) {
  if (S.hidden) return false;
  if (!S.desktop) return true;
  if (S.mode === 'grid') return true;
  return S.primary === card.id;
}

function armStall(card) {
  cancel(card.stallTimer);
  card.stallTimer = later(function () { checkStall(card); }, WATCH_MS);
}

/**
 * The watchdog proper, run every WATCH_MS while a socket is attached.
 *
 *   · a load gap past STALL_MS on a chatty engine        -> hard drop
 *   · a load gap past STALL_MS while the SERVER also says the camera is not
 *     live                                               -> hard drop
 *   · a load gap past VERIFY_MS on a quiet engine, server still happy
 *     -> silently re-open the socket. A healthy stream loses a single frame
 *        and proves itself; a dead one produces no load and falls into the
 *        drop path on the next pass. This is the only way to catch "our
 *        socket died but the pipeline is fine" on an engine that will not
 *        tell us about parts.
 */
function checkStall(card) {
  if (!card.attached) return;
  var since = Date.now() - card.lastLoad;
  var cam = card.cam || {};
  var age = num(cam.frame_age);
  var serverBad = (!!cam.state && cam.state !== 'live') ||
    (age !== null && age >= STALE_AGE);

  if (since >= STALL_MS && (CHATTY || serverBad)) {
    onStreamDrop(card, CHATTY
      ? 'No MJPEG data for ' + Math.round(since / 1000) + ' s.'
      : 'The pipeline stopped producing frames.');
    return;
  }
  if (since >= VERIFY_MS) {
    card.verifying = true;
    detachStream(card);          /* releases the old socket for certain */
    attachStream(card);          /* re-arms the watchdog on its way out */
    if (!card.attached) paintCard(card);
    return;
  }
  card.stallTimer = later(function () { checkStall(card); }, WATCH_MS);
}

function onStreamDrop(card, cause) {
  cancel(card.stallTimer); card.stallTimer = null;
  card.faultCause = cause;
  card.imgEl.removeAttribute('src');
  card.imgEl.src = BLANK;
  card.attached = false;
  scheduleReconnect(card);
}

function scheduleReconnect(card) {
  cancel(card.retryTimer);
  var delay = BACKOFF[Math.min(card.attempt, BACKOFF.length - 1)];
  card.attempt += 1;
  card.nextTryAt = Date.now() + delay;
  card.retryTimer = later(function () {
    card.retryTimer = null;
    attachStream(card);
  }, delay);
  paintCard(card);
}

function reconnectNow(card) {
  cancel(card.retryTimer);
  card.retryTimer = null;
  card.attempt = card.attempt || 1;
  card.nextTryAt = 0;
  announce('Reconnecting ' + card.name);
  attachStream(card);
}

/** The single source of truth for how a card looks. Cheap; called every tick. */
function paintCard(card) {
  var cam = card.cam || {};
  var age = num(cam.frame_age);
  var serverState = cam.state || 'unknown';
  var faulted = !!card.faultCause && !card.attached;

  var state;
  if (faulted || serverState === 'offline' || (age !== null && age >= DEAD_AGE)) {
    state = card.attempt ? 'reconnecting' : 'offline';
  } else if (serverState === 'stale' || (age !== null && age >= STALE_AGE)) {
    state = 'stale';
  } else if (!card.attached) {
    state = 'connecting';
  } else {
    state = 'live';
  }
  card.state = state;

  var hard = state === 'offline' || state === 'reconnecting';
  toggleClass(card.el, 'cam--stale', state === 'stale');
  toggleClass(card.el, 'cam--offline', hard);
  toggleClass(card.el, 'cam--no-route', serverState === 'offline' && !card.attempt);

  var ageText = age === null ? null : shortAgo(age);
  var word = state === 'live' ? 'Live'
    : state === 'stale' ? 'Stale'
      : state === 'reconnecting' ? 'Reconnecting'
        : state === 'connecting' ? 'Connecting'
          : 'Offline';
  var pillKind = state === 'live' ? 'live'
    : state === 'stale' ? 'stale'
      : state === 'reconnecting' ? 'reconnecting'
        : state === 'connecting' ? 'unknown'
          : 'offline';
  card.headStatus.set(pillKind, word, state === 'live' ? null : ageText);
  card.mediaStatus.set(pillKind, word, null);

  /* The on-media age pill — the thing that stops a frozen frame reading as
     live even when the operator only glances at the picture. */
  if (state === 'live' || age === null) {
    card.ageEl.hidden = true;
  } else {
    card.ageEl.hidden = false;
    setText(card.ageEl, 'Last frame ' + shortAgo(age) + ' ago');
  }

  setText(card.idEl, cam.location ? cam.id + ' · ' + cam.location : cam.id);

  if (cam.ptz_target) {
    card.handoff.hidden = false;
    setText(card.handoff, 'Tracking · ' + cam.id + ' hand-off');
  } else {
    card.handoff.hidden = true;
  }

  /* fault block */
  var showFault = hard || (state === 'stale' && card.attempt > 0);
  card.faultEl.hidden = !showFault;
  if (showFault) {
    setText(card.faultTitle, card.attempt ? 'Stream lost — reconnecting' : 'Stream is down');
    setText(card.faultCauseEl, card.faultCause ||
      (serverState === 'offline' ? 'The pipeline reports this camera offline.'
        : 'The stream stopped producing frames.'));
    var line;
    if (card.retryTimer && card.nextTryAt) {
      var secs = Math.max(0, Math.ceil((card.nextTryAt - Date.now()) / 1000));
      line = 'Reconnecting — attempt ' + card.attempt + ', next try in ' + secs + ' s';
    } else if (card.attempt) {
      line = 'Reconnecting — attempt ' + card.attempt + ', trying now';
    } else {
      line = 'Waiting for the pipeline to bring the camera back.';
    }
    setText(card.faultRetryEl, line);
  }

  /* controls */
  if (card.ptzEl) {
    var lock = hard;
    card.controlDown = lock;
    card.ptzEl.hidden = lock;
    card.noControl.hidden = !lock;
    if (lock) stopJog();
  } else {
    card.noControl.hidden = false;
    setText(card.noControl, 'This camera has no PTZ head.');
  }
  card.saveBtn.disabled = hard;

  /* telemetry that comes from the camera row itself */
  card.rAge.set(age === null ? '—' : shortAgo(age),
    state === 'live' ? 'live' : (state === 'stale' ? 'warn' : 'danger'));
}

/* ========================================================================
   Polling
   ===================================================================== */

function pollCameras() {
  return api.cameras({ signal: signal(), timeout: 8000 })
    .then(function (res) {
      S.camsError = null;
      S.tz = (res && res.timezone) || '';
      var rows = (res && res.cameras) || [];
      S.cams = rows;
      store.set({ cameras: rows });
      syncCards(rows);
      renderSummary();
    })
    .catch(function (err) {
      if (api.isAbort(err)) return;
      S.camsError = err;
      renderSummary();
      if (!S.cards.size) renderFallback();
      shout('cameras', 'Lost the camera list.', err, {
        retry: function () { pollCameras(); }
      });
    });
}

function pollMonitor() {
  return api.monitor({ signal: signal(), timeout: 8000 })
    .then(function (res) {
      var rows = (res && res.cameras) || [];
      var byId = Object.create(null);
      rows.forEach(function (r) { byId[r.id] = r; });
      S.monitor = byId;
      S.cards.forEach(function (card) { paintTelemetry(card); });
    })
    .catch(function (err) {
      if (api.isAbort(err)) return;
      S.cards.forEach(function (card) {
        card.rBuffer.set('—', 'danger', 0, 'buffer unavailable');
        card.rTracks.set('—', 'danger');
      });
      shout('monitor', 'Pipeline telemetry is unavailable.', err);
    });
}

function paintTelemetry(card) {
  var m = S.monitor[card.id];
  if (!m) {
    card.rBuffer.set('—', null, 0, 'buffer unknown');
    card.rTracks.set('—');
    card.rTop.set('—', null, 0, 'no detection');
    card.rEvent.set('—');
    return;
  }
  var secs = num(m.buffer_seconds);
  var maxSecs = num(m.buffer_max_seconds) || 30;
  var lit = secs === null ? 0 : Math.round(clamp(secs / maxSecs, 0, 1) * 10);
  var bufMod = secs === null ? null : (secs < maxSecs * 0.15 ? 'warn' : 'live');
  card.rBuffer.set(secs === null ? '—' : secs.toFixed(1), bufMod, lit,
    'buffer ' + (secs === null ? 'unknown' : secs.toFixed(1) + ' of ' + maxSecs + ' seconds'));

  var tracks = num(m.tracks_active);
  card.rTracks.set(tracks === null ? '—' : String(tracks), tracks ? 'live' : null);

  var conf = num(m.event_confidence);
  card.rTop.set(conf === null ? '—' : conf.toFixed(2),
    conf === null ? null : (conf < 0.5 ? 'warn' : 'live'),
    conf === null ? 0 : Math.round(clamp(conf, 0, 1) * 8),
    'top confidence ' + (conf === null ? 'none' : conf.toFixed(2)));

  var dur = num(m.event_duration);
  var species = (m.event_species && m.event_species.length)
    ? String(m.event_species[0]).split('_').pop() : null;
  if (m.event_active && dur !== null) {
    card.rEvent.set(dur.toFixed(1) + 's', 'live');
    showReticle(card, species, conf);
  } else {
    card.rEvent.set('idle');
    card.reticleEl.hidden = true;
  }

  if (m.status && m.status !== 'connected') {
    card.faultCause = card.faultCause || ('Pipeline reports: ' + m.status);
  }
}

function showReticle(card, species, conf) {
  /* The pipeline gives us the fact of a detection, not its box. The reticle
     therefore marks the centre third and names the track — it is a "this
     frame is under machine control" marker, never a fake bounding box. */
  card.reticleEl.hidden = false;
  card.reticleEl.style.setProperty('--x', '33%');
  card.reticleEl.style.setProperty('--y', '33%');
  card.reticleEl.style.setProperty('--w', '34%');
  card.reticleEl.style.setProperty('--h', '34%');
  var bits = [];
  if (species) bits.push(species);
  if (conf !== null && conf !== undefined) bits.push(conf.toFixed(2));
  setText(card.reticleLabel, 'TRK · ' + (bits.length ? bits.join(' ') : 'active'));
}

function pollPositions() {
  S.cards.forEach(function (card) {
    if (!card.cam.has_ptz) return;
    if (!cardWantsStream(card)) return;
    if (card.posInFlight) return;
    card.posInFlight = true;
    api.ptz.position(card.ptzId, { signal: signal(), timeout: 6000 })
      .then(function (pos) {
        card.posInFlight = false;
        card.position = pos;
        card.posError = null;
        paintPosition(card);
      })
      .catch(function (err) {
        card.posInFlight = false;
        if (api.isAbort(err)) return;
        card.posError = err;
        paintPosition(card);
      });
  });
}

function paintPosition(card) {
  if (!card.readoutEl) return;
  var p = card.position;
  if (card.posError || !p || p.available === false) {
    setText(card.pEl, '—');
    setText(card.tEl, '—');
    setText(card.zEl, '—');
    setText(card.posNote, card.posError
      ? 'position unavailable — ' + api.describe(card.posError)
      : 'this head does not report its position');
    card.readoutEl.setAttribute('aria-label', 'PTZ position unavailable');
    return;
  }
  setText(card.pEl, fixed(p.pan));
  setText(card.tEl, fixed(p.tilt));
  setText(card.zEl, fixed(p.zoom));
  setText(card.posNote, '');
  card.readoutEl.setAttribute('aria-label',
    'Pan ' + fixed(p.pan) + ', tilt ' + fixed(p.tilt) + ', zoom ' + fixed(p.zoom));
  var z = num(p.zoom);
  if (z !== null && card.zoomSlider && !card.zoomSlider.isDragging()) {
    var v = clamp(z < 0 ? (z + 1) / 2 : z, 0, 1);
    card.zoomSlider.set(v);
    setText(card.zoomOutput, Math.round(v * 100) + '%');
  }
}

function loadPresets(card, force) {
  if (!card.cam.has_ptz) return;
  if (card.presetsLoaded && !force) return;
  card.presetsLoaded = true;
  api.ptz.presets(card.ptzId, { signal: signal(), timeout: 8000 })
    .then(function (res) {
      var list = (res && res.presets) || [];
      card.presets = list;
      card.presetsError = (res && res.error) || null;
      renderPresets(card);
    })
    .catch(function (err) {
      if (api.isAbort(err)) return;
      card.presets = [];
      card.presetsError = api.describe(err);
      renderPresets(card);
    });
}

function renderPresets(card) {
  var items = card.presets.filter(function (p) { return p && p.token; });
  keyedList(card.presetsEl, items, {
    key: function (p) { return String(p.token); },
    create: function () {
      return h('button.chip', { type: 'button' }, h('span'));
    },
    update: function (el, p, key, i) {
      el.setAttribute('data-preset', p.token);
      el.setAttribute('data-name', p.name || p.token);
      setText(el.firstChild, (i < 9 ? (i + 1) + ' · ' : '') + (p.name || p.token));
      el.setAttribute('aria-label', 'Recall preset ' + (p.name || p.token) +
        (i < 9 ? ' (key ' + (i + 1) + ')' : ''));
    }
  });
  card.presetsEl.appendChild(card.savePresetBtn);
  if (card.presetsError) {
    card.presetNote.hidden = false;
    card.presetNote.classList.add('t-danger');
    setText(card.presetNote, 'Presets unavailable — ' + card.presetsError);
  } else if (!items.length) {
    card.presetNote.hidden = false;
    card.presetNote.classList.remove('t-danger');
    setText(card.presetNote, 'No presets stored on this head yet.');
  } else {
    card.presetNote.hidden = true;
  }
}

function loadMode(card) {
  if (!card.cam.has_ptz) return;
  api.ptz.mode(card.ptzId, { signal: signal(), timeout: 8000 })
    .then(function (m) {
      card.mode = m;
      card.trackSwitch.enable();
      card.patrolSwitch.enable();
      card.trackSwitch.set(!!m.track_enabled, false);
      card.patrolSwitch.set(!!m.patrol_enabled, false);
      applyTrackBanner(card, !!m.track_enabled);
      var d = num(m.patrol_return_delay);
      if (d !== null) {
        card.returnSlider.set(d);
        setText(card.returnOut, d + ' s');
      }
    })
    .catch(function (err) {
      if (api.isAbort(err)) return;
      card.trackSwitch.disable('Unknown');
      card.patrolSwitch.disable('Unknown');
      shout('mode-' + card.ptzId, 'Could not read the PTZ mode for ' + card.ptzId + '.', err);
    });
}

function loadDebug(card) {
  api.ptz.debug({ signal: signal(), timeout: 8000 })
    .then(function (res) {
      card.debugSwitch.set(!!(res && res.enabled), false);
    })
    .catch(function () {
      card.debugSwitch.disable('Unknown');
    });
}

/* ========================================================================
   Card list, filmstrip and layout
   ===================================================================== */

function syncCards(rows) {
  var seen = Object.create(null);
  var built = false;
  rows.forEach(function (cam) {
    seen[cam.id] = true;
    var card = S.cards.get(cam.id);
    if (!card) {
      card = buildCard(cam);
      S.cards.set(cam.id, card);
      built = true;
      if (cam.has_ptz) {
        loadPresets(card);
        loadMode(card);
        loadDebug(card);
      }
    } else {
      card.cam = cam;
      card.ptzId = ptzIdFor(cam);
    }
    paintCard(card);
    paintTelemetry(card);
  });

  S.cards.forEach(function (card, id) {
    if (seen[id]) return;
    detachStream(card);
    if (card.el.parentNode) card.el.parentNode.removeChild(card.el);
    S.cards.delete(id);
  });

  if (!S.primary || !seen[S.primary]) S.primary = rows.length ? rows[0].id : null;
  applyLayout();
  /* A new card would otherwise read "-" for pan/tilt/zoom until the position
     poll next comes round, which looks like "this head has no encoder". */
  if (built) pollPositions();
}

function buildStripItem(cam) {
  var img = h('img.frame__img', { alt: '', decoding: 'async' });
  var dot = h('span.camrow__dot');
  var label = h('span.camstrip__label', dot, h('span.truncate'));
  var btn = h('button.camstrip__item', { type: 'button' },
    h('div.frame.frame--sm', h('div.frame__film.film--day'), img), label);
  btn._img = img;
  btn._dot = dot;
  btn._name = label.lastChild;
  return btn;
}

function renderStrip(others) {
  keyedList(S.strip, others, {
    key: function (cam) { return cam.id; },
    create: function (cam) { return buildStripItem(cam); },
    update: function (el, cam) {
      setText(el._name, cam.name || cam.id);
      el.setAttribute('aria-pressed', 'false');
      el.setAttribute('aria-label', 'Show ' + (cam.name || cam.id) + ' on the main stage');
      el._dot.className = 'camrow__dot camrow__dot--' +
        (cam.state === 'live' ? 'live' : cam.state === 'stale' ? 'stale' : 'offline');
      if (!el._img.getAttribute('src')) refreshSnapshot(el._img, cam.id);
    }
  });
}

function refreshSnapshot(img, id) {
  if (S.hidden) return;
  img.src = '/snapshot/' + encodeURIComponent(id) + '?_=' + Date.now();
}

function refreshSnapshots() {
  if (S.hidden || !S.desktop || S.mode !== 'focus') return;
  var child = S.strip.firstElementChild;
  while (child) {
    if (child._img) refreshSnapshot(child._img, child.getAttribute('data-key'));
    child = child.nextElementSibling;
  }
}

/**
 * Mobile: every camera is a full card in one column.
 * Desktop focus: primary stage + 320px operator column + filmstrip.
 * Desktop grid: every camera as a full card, two up.
 *
 * The card DOM is built once per camera; layout only re-parents nodes, so
 * listeners, focus, decoded frames and in-flight gestures all survive.
 */
/**
 * With no cards there is nothing for the layout to place, and the boot
 * skeleton would sit there for ever. A failed camera list has to say so in
 * the page, not only in a toast that has already timed out.
 */
function renderFallback() {
  var kind = S.camsError ? 'error' : 'empty';
  if (S.fallbackKind === kind && S.fallbackEl && S.fallbackEl.parentNode === S.wall) {
    if (kind === 'error' && S.fallbackBody) {
      setText(S.fallbackBody, api.describe(S.camsError));
    }
    return;
  }
  var el;
  if (kind === 'error') {
    var retryBtn = h('button.btn.btn--primary', { type: 'button' },
      icon('refresh', { size: 'sm', 'class': 'btn__icon' }),
      h('span.btn__label', 'Try again'));
    retryBtn.addEventListener('click', function () { pollCameras(); });
    S.fallbackBody = h('p.empty__body', { text: api.describe(S.camsError) });
    el = h('div.empty.empty--error', { role: 'alert' },
      h('div.empty__art', icon('alert', { size: 'lg' })),
      h('h2.empty__title', 'The camera list did not load'),
      S.fallbackBody,
      h('p.empty__endpoint', 'GET /api/cameras'),
      h('div.empty__actions', retryBtn));
  } else {
    S.fallbackBody = null;
    el = h('div.empty',
      h('div.empty__art', icon('camera', { size: 'lg' })),
      h('h2.empty__title', 'No cameras are configured'),
      h('p.empty__body',
        'The pipeline reported an empty camera list. Add one to ' +
        'config/cameras.yml and restart the service.'));
  }
  S.fallbackEl = el;
  S.fallbackKind = kind;
  placeChildren(S.wall, [el]);
}

function applyLayout() {
  if (!S.wall) return;
  if (!S.cards.size) { renderFallback(); return; }
  S.fallbackEl = null;
  S.fallbackKind = null;
  var rows = S.cams;
  var desktopFocus = S.desktop && S.mode === 'focus';

  S.wall.classList.toggle('camwall--focus', desktopFocus);
  S.wall.classList.toggle('camwall--grid', S.desktop && S.mode === 'grid');

  var order = [];
  if (desktopFocus) {
    var prim = S.cards.get(S.primary);
    if (prim) order.push(prim);
  } else {
    rows.forEach(function (cam) {
      var c = S.cards.get(cam.id);
      if (c) order.push(c);
    });
  }

  /* Place the cards. This runs on every camera poll, so it must move only
     what is genuinely out of place: clearing and re-appending would tear
     down a running MJPEG socket and blur whatever had focus, three seconds
     at a time. */
  var nodes = [];
  S.cards.forEach(function (card) { card.el.classList.remove('cam--primary'); });
  order.forEach(function (card) {
    if (desktopFocus) card.el.classList.add('cam--primary');
    nodes.push(card.el);
  });
  if (desktopFocus) {
    nodes.push(S.operator);
    nodes.push(S.strip);
  }
  placeChildren(S.wall, nodes);

  /* Re-parent the operator pieces. */
  S.cards.forEach(function (card) {
    var wantsOperator = desktopFocus && card.id === S.primary;
    var host = wantsOperator ? S.operator : card.bodyEl;
    [card.telemetryEl, card.saveBtn, card.ptzEl].forEach(function (node) {
      if (!node) return;
      if (node.parentNode !== host) host.appendChild(node);
    });
    if (!wantsOperator && card.faultEl.parentNode !== card.bodyEl) {
      card.bodyEl.appendChild(card.faultEl);
    }
    /* The fault block stays in the card on every layout — that was the whole
       point of the spec's "never desktop-only" note. */
    if (card.bodyEl.firstChild !== card.faultEl && card.faultEl.parentNode === card.bodyEl) {
      card.bodyEl.insertBefore(card.faultEl, card.bodyEl.firstChild);
    }
    if (card.noControl.parentNode !== card.bodyEl) card.bodyEl.appendChild(card.noControl);
  });

  if (desktopFocus) {
    var others = rows.filter(function (c) { return c.id !== S.primary; });
    S.strip.hidden = others.length === 0;
    renderStrip(others);
  } else {
    S.strip.hidden = true;
    clear(S.strip);
  }

  /* Streams follow the layout: only visible stages hold a socket. */
  S.cards.forEach(function (card) {
    if (cardWantsStream(card)) {
      if (!card.attached && !card.retryTimer) attachStream(card);
    } else if (card.attached) {
      detachStream(card);
      paintCard(card);
    }
  });

  if (S.layoutBtn) {
    var gridOn = S.mode === 'grid';
    S.layoutBtn.setAttribute('aria-pressed', gridOn ? 'true' : 'false');
    setText(S.layoutBtn.lastChild, gridOn ? 'Focus' : 'Grid');
    S.layoutBtn.hidden = !S.desktop;
  }
}

function setPrimary(id) {
  if (S.primary === id) return;
  S.primary = id;
  applyLayout();
  var card = S.cards.get(id);
  if (card) {
    card.stageEl.focus();
    announce(card.name + ' promoted to the main stage');
  }
  /* setQuery, not go(): it merges rather than replacing the query, and the
     router hands the change to update() instead of remounting the view —
     which would kill every stream and rebuild every card. */
  router.setQuery({ camera: id }, { replace: true });
}

/* ========================================================================
   Chrome, summary and the streams sheet
   ===================================================================== */

function summarise() {
  var counts = { live: 0, stale: 0, offline: 0 };
  S.cards.forEach(function (card) {
    var st = card.state;
    if (st === 'live') counts.live += 1;
    else if (st === 'stale') counts.stale += 1;
    else if (st === 'offline' || st === 'reconnecting') counts.offline += 1;
  });
  var bits = [];
  if (counts.live) bits.push(counts.live + ' live');
  if (counts.stale) bits.push(counts.stale + ' stale');
  if (counts.offline) bits.push(counts.offline + ' down');
  if (!bits.length) bits.push(plural(S.cams.length, 'camera', 'cameras'));
  return { text: bits.join(' · '), counts: counts };
}

function renderSummary() {
  var s = summarise();
  var subtitle = S.camsError ? 'Camera list unavailable' : s.text;
  /* Called every second by tick(); only a real change touches the chrome or
     the live region, or a screen reader would never stop talking. */
  if (subtitle !== S.lastSummary) {
    S.lastSummary = subtitle;
    store.setChrome({ subtitle: subtitle });
    announce('Cameras: ' + subtitle);
  }
  if (S.streamsBtn) {
    setText(S.streamsBtn.lastChild, s.counts.offline || s.counts.stale ? s.text : 'Streams');
    S.streamsBtn.classList.toggle('btn--danger', s.counts.offline > 0);
  }
}

function openStreamsSheet() {
  var body = h('div.stack');
  sheet({
    title: 'Streams',
    snap: 'half',
    content: function (host) {
      host.appendChild(body);
    }
  });
  var rows = S.cams;
  rows.forEach(function (cam) {
    var card = S.cards.get(cam.id);
    if (!card) return;
    var age = num(cam.frame_age);
    var pill = makeStatusPill(false);
    var word = card.state === 'live' ? 'Live'
      : card.state === 'stale' ? 'Stale'
        : card.state === 'reconnecting' ? 'Reconnecting' : 'Offline';
    pill.set(card.state === 'live' ? 'live'
      : card.state === 'stale' ? 'stale'
        : card.state === 'reconnecting' ? 'reconnecting' : 'offline',
      word, age === null ? null : shortAgo(age));
    var btn = h('button.btn.btn--secondary.btn--sm', { type: 'button' },
      icon('refresh', { size: 'sm', 'class': 'btn__icon' }),
      h('span.btn__label', 'Reconnect'));
    btn.addEventListener('click', function () { reconnectNow(card); });
    body.appendChild(h('div.row.row--between',
      h('div.stack.stack--tight',
        h('span', { text: cam.name || cam.id }),
        h('span.t-xs.t-3', {
          text: (cam.location ? cam.location + ' · ' : '') +
            (age === null ? 'no frame age reported' : 'last frame ' + shortAgo(age) + ' ago')
        })),
      h('div.row', pill.el, btn)));
  });
  if (!rows.length) {
    body.appendChild(h('p.t-sm.t-3', { text: 'The server reported no cameras.' }));
  }
}

/* ========================================================================
   Visibility, global stops
   ===================================================================== */

function syncVisibility() {
  var hidden = document.visibilityState === 'hidden';
  if (hidden === S.hidden) return;
  S.hidden = hidden;
  /* A jog must never survive the tab going away. */
  stopJog();
  if (hidden) {
    S.cards.forEach(function (card) { detachStream(card); });
    stopPolling();
  } else {
    startPolling();
    S.cards.forEach(function (card) {
      card.attempt = 0;
      card.faultCause = '';
      if (cardWantsStream(card)) attachStream(card);
    });
  }
}

function stopPolling() {
  cancel(S.camTimer); S.camTimer = null;
  cancel(S.monTimer); S.monTimer = null;
  cancel(S.posTimer); S.posTimer = null;
  cancel(S.snapTimer); S.snapTimer = null;
}

function startPolling() {
  stopPolling();
  pollCameras();
  pollMonitor();
  pollPositions();
  S.camTimer = every(pollCameras, CAMERAS_MS);
  S.monTimer = every(pollMonitor, MONITOR_MS);
  S.posTimer = every(pollPositions, POSITION_MS);
  S.snapTimer = every(refreshSnapshots, SNAPSHOT_MS);
}

function tick() {
  var now = new Date();
  var clock = wallClock(now);
  S.cards.forEach(function (card) {
    setText(card.clockEl, clock);
    /* frame_age advances between polls; the pill must not sit still. */
    if (card.cam && num(card.cam.frame_age) !== null && !S.hidden) {
      card.cam.frame_age = card.cam.frame_age + TICK_MS / 1000;
    }
    paintCard(card);
    var m = S.monitor[card.id];
    if (m) {
      setText(card.rateEl, (num(m.buffer_frames) || 0) + ' fr buffered');
    }
  });
  renderSummary();
}

/* ========================================================================
   The view
   ===================================================================== */

export const view = {
  mount: function (root, ctx) {
    S = newState();
    S.root = root;

    var q = (ctx && ctx.query) || {};
    if (q.camera) S.primary = String(q.camera);
    if (q.mode === 'grid') S.mode = 'grid';

    /* --- chrome -------------------------------------------------------- */
    S.layoutBtn = h('button.btn.btn--secondary.btn--sm', {
      type: 'button', 'aria-pressed': 'false'
    }, icon('grid', { size: 'sm', 'class': 'btn__icon' }), h('span.btn__label', 'Grid'));
    S.layoutBtn.addEventListener('click', function () {
      S.mode = S.mode === 'grid' ? 'focus' : 'grid';
      applyLayout();
      router.setQuery({ mode: S.mode === 'grid' ? 'grid' : null }, { replace: true });
    });

    S.streamsBtn = h('button.btn.btn--secondary.btn--sm', { type: 'button' },
      icon('live', { size: 'sm', 'class': 'btn__icon' }),
      h('span.btn__label', 'Streams'));
    S.streamsBtn.addEventListener('click', openStreamsSheet);

    /* The buttons live in the view, not in chrome.actions: .topbar is
       display:none from 1024px up, and Grid is a desktop-only affordance —
       parking it there would hide it exactly where it is needed. */
    store.setChrome({
      title: 'Live',
      subtitle: 'Connecting…',
      actions: [],
      toolbar: null,
      rail: null,
      norail: true,
      selbar: null,
      mods: ['no-blur']
    });

    /* --- skeleton ------------------------------------------------------ */
    root.appendChild(h('h1.visually-hidden', { tabIndex: -1, text: 'Live cameras' }));
    S.liveRegion = h('p.visually-hidden', {
      'aria-live': 'polite', 'aria-atomic': 'true'
    });
    root.appendChild(S.liveRegion);

    S.toolbar = h('div.row.row--between', { role: 'group', 'aria-label': 'Live controls' },
      h('span.overline.overline--strong', { text: 'Cameras' }),
      h('div.row', S.layoutBtn, S.streamsBtn));
    root.appendChild(S.toolbar);

    S.wall = h('div.camwall');
    S.operator = h('div.stack', { role: 'group', 'aria-label': 'Operator controls' });
    S.strip = h('div.camstrip', { role: 'group', 'aria-label': 'Other cameras' });
    S.strip.hidden = true;
    root.appendChild(S.wall);

    reg(delegate(S.strip, 'click', '.camstrip__item', function (ev, node) {
      var id = node.getAttribute('data-key');
      if (id) setPrimary(id);
    }));

    /* placeholder while the first poll is in flight */
    S.wall.appendChild(h('div.cam',
      h('div.cam__stage', h('div.frame.frame--skel')),
      h('div.cam__body', h('div.skel.skel--title'), h('div.skel.skel--row'))));

    /* --- responsive ---------------------------------------------------- */
    if (window.matchMedia) {
      S.mq = window.matchMedia('(min-width: 1024px)');
      S.desktop = S.mq.matches;
      var onMq = function () {
        S.desktop = S.mq.matches;
        S.layoutBtn.hidden = !S.desktop;
        applyLayout();
      };
      S.layoutBtn.hidden = !S.desktop;
      if (S.mq.addEventListener) {
        S.mq.addEventListener('change', onMq);
        S.mqOff = function () { S.mq.removeEventListener('change', onMq); };
      } else if (S.mq.addListener) {
        S.mq.addListener(onMq);
        S.mqOff = function () { S.mq.removeListener(onMq); };
      }
      reg(S.mqOff);
    }

    /* --- global stop guarantees --------------------------------------- */
    reg(on(document, 'visibilitychange', syncVisibility));
    reg(on(window, 'blur', function () { stopJog(); }));
    reg(on(window, 'pagehide', function () { stopJog(); }));
    reg(on(document, 'keydown', function (ev) {
      if (ev.key === 'Escape') stopJog();
    }, true));
    /* A pointer released anywhere — outside the button, over the scrim,
       past the window edge — still ends the jog. */
    reg(on(window, 'pointerup', function () { stopJog(); }));
    reg(on(window, 'pointercancel', function () { stopJog(); }));
    reg(on(window, 'touchcancel', function () { stopJog(); }));

    /* The shell already holds a camera list for its rail; painting from it
       first means the operator sees stages, not a skeleton, while the first
       poll is still in flight. */
    var known = store.get('cameras');
    if (known && known.length) {
      S.cams = known;
      syncCards(known);
      renderSummary();
    }

    S.tickTimer = every(tick, TICK_MS);
    startPolling();

    return true;
  },

  /* The router's contract for a same-route query change. Without it a Back
     press, or the Grid toggle's own setQuery, would unmount and remount the
     whole surface — tearing down every MJPEG socket, every poll and any jog
     in flight. ?camera= and ?mode= are the only state the URL carries. */
  update: function (ctx) {
    if (!S) return;
    var q = (ctx && ctx.query) || {};
    var mode = q.mode === 'grid' ? 'grid' : 'focus';
    var camera = q.camera ? String(q.camera) : null;
    var changed = false;
    if (mode !== S.mode) { S.mode = mode; changed = true; }
    if (camera && camera !== S.primary && S.cards.has(camera)) {
      S.primary = camera;
      changed = true;
    }
    if (changed) applyLayout();
  },

  unmount: function () {
    if (!S) return;
    S.dead = true;
    stopJog();
    stopPolling();
    cancel(S.tickTimer);

    S.cards.forEach(function (card) {
      cancel(card.retryTimer);
      cancel(card.stallTimer);
      cancel(card.pulseTimer);
      cancel(card.crossTimer);
      detachStream(card);
    });

    for (var i = 0; i < S.timers.length; i++) {
      window.clearTimeout(S.timers[i]);
      window.clearInterval(S.timers[i]);
    }
    S.timers.length = 0;

    for (var d = 0; d < S.disposers.length; d++) {
      try { S.disposers[d](); } catch (e) { /* a disposer must not block the rest */ }
    }
    S.disposers.length = 0;

    if (S.aborter) { try { S.aborter.abort(); } catch (e) { /* older WebKit */ } }
    S.cards.clear();
    if (S.root) clear(S.root);
    S = null;
  }
};

export default view;
